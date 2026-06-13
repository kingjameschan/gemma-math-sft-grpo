"""Gold PPL evolution: R16 GRPO LoRA ckpts at step 6, 32, 58.
   step 42 already exists at r16_step42.json — skip.

Chunked save: per_q records saved every 50 batches to .json so partial data survives crash/preempt.
"""
import json, time, math, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path('/mnt/d/fine-tuning')
MODEL_PATH = ROOT / 'models/gemma-2-2b-it'
TEST_PC = ROOT / 'v3/shared/data/gsm8k/test_pc.jsonl'
OUT_DIR = ROOT / 'v3/E5_grpo/outputs/gold_ppl'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R16_BASE = ROOT / 'v3/E5_grpo/checkpoints/baseit_r16_clean_grpo_eval_root/baseit_r16_clean_grpo'

# Steps to eval (skip 42 — already done)
STEPS = [6, 32, 58]
if len(sys.argv) > 1:
    STEPS = [int(x) for x in sys.argv[1].split(',')]
print(f'[plan] STEPS={STEPS}')

device = 'cuda'
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
data = [json.loads(l) for l in open(TEST_PC)]
print(f'[data] n_test={len(data)}')

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

indexed = sorted(enumerate(seqs), key=lambda kv: -kv[1]['full_len'])
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
BS = 4

# Load base ONCE
print('[load] base model bf16...')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='eager'
)

for step in STEPS:
    LORA = R16_BASE / f'global_step_{step}' / 'actor' / 'lora_adapter'
    OUT_FILE = OUT_DIR / f'r16_step{step}.json'

    if not LORA.exists():
        print(f'[skip] step {step}: LoRA dir missing {LORA}')
        continue

    print(f'\n=== R16 step {step} ===')
    print(f'[load] LoRA from {LORA}')
    model = PeftModel.from_pretrained(base_model, str(LORA))
    model = model.merge_and_unload()
    model.eval()

    per_q_records = {}
    sum_nll, sum_tok = 0.0, 0
    t0 = time.time()
    processed = 0
    SAVE_EVERY_BATCHES = 50

    with torch.inference_mode():
        for batch_start in range(0, len(indexed), BS):
            batch = indexed[batch_start:batch_start+BS]
            max_len = max(s['full_len'] for _, s in batch)
            B = len(batch)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            for i, (_, s) in enumerate(batch):
                input_ids[i, :s['full_len']] = torch.tensor(s['full'], device=device)
            attn = (input_ids != pad_id).long()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
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
            batch_i = batch_start // BS
            if batch_i % SAVE_EVERY_BATCHES == 0 and batch_i > 0:
                # CHUNKED SAVE: dump partial each 50 batches
                partial = {
                    'target': f'r16_step{step}', 'n_q': len(per_q_records),
                    'mean_nll_per_token': sum_nll / sum_tok if sum_tok else None,
                    'mean_ppl': math.exp(sum_nll / sum_tok) if sum_tok else None,
                    'total_nll': sum_nll, 'total_tokens': sum_tok,
                    'elapsed_s': time.time() - t0, 'incomplete': True,
                    'per_q': [per_q_records[k] for k in sorted(per_q_records.keys())],
                }
                json.dump(partial, open(str(OUT_FILE) + '.partial', 'w'))
                elapsed = time.time() - t0
                print(f'  [step{step}] {processed}/{len(seqs)} ({processed/len(seqs)*100:.0f}%)  '
                      f'PPL={math.exp(sum_nll/sum_tok):.3f}  elapsed={elapsed:.0f}s  [partial saved]', flush=True)
                torch.cuda.empty_cache()

    mean_nll = sum_nll / sum_tok
    per_q = [per_q_records[k] for k in sorted(per_q_records.keys())]
    out = {
        'target': f'r16_step{step}', 'n_q': len(per_q),
        'mean_nll_per_token': mean_nll, 'mean_ppl': math.exp(mean_nll),
        'total_nll': sum_nll, 'total_tokens': sum_tok,
        'elapsed_s': time.time() - t0, 'per_q': per_q,
    }
    json.dump(out, open(OUT_FILE, 'w'))
    # cleanup .partial
    if Path(str(OUT_FILE) + '.partial').exists():
        Path(str(OUT_FILE) + '.partial').unlink()
    print(f'[DONE step {step}] PPL={out["mean_ppl"]:.3f} saved → {OUT_FILE}')

    # Free for next ckpt
    del model
    torch.cuda.empty_cache()
    # Reload base for next iter (since merge_and_unload mutates base_model)
    if step != STEPS[-1]:
        print('[reload] reloading base for next ckpt...')
        del base_model
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='eager'
        )

print('\n=== ALL DONE ===')
for step in STEPS:
    f = OUT_DIR / f'r16_step{step}.json'
    if f.exists():
        d = json.load(open(f))
        print(f'  step {step}: PPL={d["mean_ppl"]:.3f}  n_q={d["n_q"]}')
