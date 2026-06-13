"""Gold PPL for SFT lr5e-4 ck-130 only (incremental, base/R15/R16 already done)."""
import json, time, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path('/mnt/d/fine-tuning')
MODEL_PATH = ROOT / 'models/gemma-2-2b-it'
TEST_PC = ROOT / 'v3/shared/data/gsm8k/test_pc.jsonl'
OUT_DIR = ROOT / 'v3/E5_grpo/outputs/gold_ppl'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SFT_LORA = ROOT / 'v3/E2_sft/checkpoints/sft_lr5e-4_r64/checkpoint-130'

device = 'cuda'
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
data = [json.loads(l) for l in open(TEST_PC)]
print(f'n_test={len(data)}')

# Same prep as _gold_ppl.py
chat_texts = []
for d in data:
    text = tok.apply_chat_template([d['prompt'][0], d['completion'][0]], tokenize=False, add_generation_prompt=False)
    prompt_text = tok.apply_chat_template([d['prompt'][0]], tokenize=False, add_generation_prompt=True)
    chat_texts.append((text, prompt_text))
seqs = []
for full_text, prompt_text in chat_texts:
    full_ids = tok(full_text, return_tensors=None)['input_ids']
    prompt_ids = tok(prompt_text, return_tensors=None)['input_ids']
    seqs.append({'full': full_ids, 'prompt_len': len(prompt_ids), 'full_len': len(full_ids)})

# Sorted descending by full_len
indexed = sorted(enumerate(seqs), key=lambda kv: -kv[1]['full_len'])

print('[load] base + LoRA merge SFT lr5e-4 ck-130...')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='eager'
)
sft = PeftModel.from_pretrained(base_model, str(SFT_LORA))
sft = sft.merge_and_unload()
sft.eval()

per_q_records = {}
sum_nll, sum_tok = 0.0, 0
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
BS = 4
t0 = time.time()
processed = 0
with torch.inference_mode():
    for batch_start in range(0, len(indexed), BS):
        batch = indexed[batch_start:batch_start+BS]
        max_len = max(s['full_len'] for _, s in batch)
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        for i, (_, s) in enumerate(batch):
            input_ids[i, :s['full_len']] = torch.tensor(s['full'], device=device)
        attn = (input_ids != pad_id).long()
        logits = sft(input_ids=input_ids, attention_mask=attn).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_targets = input_ids[:, 1:].contiguous()
        nll_per_pos = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_targets.reshape(-1), reduction='none',
        ).reshape(shifted_targets.shape)
        del logits, shifted_logits, shifted_targets
        for i, (orig_idx, s) in enumerate(batch):
            start = max(s['prompt_len'] - 1, 0)
            end = s['full_len'] - 1
            if end <= start: continue
            seq_nll = nll_per_pos[i, start:end].sum().item()
            n_tok = end - start
            per_q_records[orig_idx] = {
                'q_idx': orig_idx, 'nll_sum': seq_nll, 'n_tok': n_tok,
                'ppl': math.exp(seq_nll / n_tok) if n_tok else float('inf'),
            }
            sum_nll += seq_nll; sum_tok += n_tok
        processed += B
        del nll_per_pos
        if (batch_start // BS) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  [sft_ck130] {processed}/{len(seqs)} ({processed/len(seqs)*100:.0f}%)  '
                  f'cum NLL/tok={sum_nll/sum_tok:.4f}  PPL={math.exp(sum_nll/sum_tok):.3f}  elapsed={elapsed:.0f}s', flush=True)
            torch.cuda.empty_cache()

mean_nll = sum_nll / sum_tok
per_q = [per_q_records[i] for i in sorted(per_q_records.keys())]
out = {
    'target': 'sft_ck130', 'n_q': len(per_q),
    'mean_nll_per_token': mean_nll, 'mean_ppl': math.exp(mean_nll),
    'total_nll': sum_nll, 'total_tokens': sum_tok,
    'elapsed_s': time.time() - t0, 'per_q': per_q,
}
json.dump(out, open(OUT_DIR / 'sft_ck130.json', 'w'))
print(f"saved → {OUT_DIR / 'sft_ck130.json'}  PPL={out['mean_ppl']:.3f}")
