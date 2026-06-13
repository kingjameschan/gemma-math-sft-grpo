"""Compute gold-answer NLL/PPL on GSM8K test (n=1319) for base + R15 ck-15 + R16 step_42.

For each (prompt, gold_completion) pair:
  - Apply Gemma2 chat template
  - Forward pass full sequence
  - Compute -log P(gold_token | prev) summed over completion tokens only
  - Aggregate per-Q NLL (sum) and n_tokens; mean_NLL_per_token = Σ nll / Σ tokens
  - PPL = exp(mean_NLL_per_token)
"""
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

TARGETS = {
    'base':      None,
    'r15_ck15':  ROOT / 'v3/E5_grpo/checkpoints/baseit_r15_verl_dapo_full_15ep_eval_root/r15_dapo/checkpoint-15',
    'r16_step42': ROOT / 'v3/E5_grpo/checkpoints/baseit_r16_clean_grpo_eval_root/baseit_r16_clean_grpo/global_step_42/actor/lora_adapter',
}

device = 'cuda'
print(f'[load] tokenizer + test_pc...')
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
data = [json.loads(l) for l in open(TEST_PC)]
print(f'  n_test = {len(data)}')

# Build chat-format input_ids and label mask (completion tokens only)
print('[prep] tokenizing all (prompt + completion) sequences...')
chat_texts = []
for d in data:
    text = tok.apply_chat_template(
        [d['prompt'][0], d['completion'][0]],
        tokenize=False, add_generation_prompt=False
    )
    # Also get prompt-only length for masking
    prompt_text = tok.apply_chat_template(
        [d['prompt'][0]], tokenize=False, add_generation_prompt=True
    )
    chat_texts.append((text, prompt_text))

# Pre-tokenize: full sequence + prompt-only (prefix len)
seqs = []
for full_text, prompt_text in chat_texts:
    full_ids = tok(full_text, return_tensors=None)['input_ids']
    prompt_ids = tok(prompt_text, return_tensors=None)['input_ids']
    seqs.append({'full': full_ids, 'prompt_len': len(prompt_ids), 'full_len': len(full_ids)})
print(f'  mean full_len={sum(s["full_len"] for s in seqs)/len(seqs):.0f}  '
      f'mean comp_tokens={sum(s["full_len"]-s["prompt_len"] for s in seqs)/len(seqs):.0f}')


def compute_ppl(model, target_name):
    model.eval()
    per_q_records = {}  # idx → record (so we can re-sort to original order later)
    sum_nll, sum_tok = 0.0, 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # Sort by full_len descending → similar-length items in same batch → less padding waste
    indexed = sorted(enumerate(seqs), key=lambda kv: -kv[1]['full_len'])
    BS = 4  # halved from 8 for 5080 16GB safety
    t0 = time.time()
    processed = 0
    with torch.inference_mode():
        for batch_start in range(0, len(indexed), BS):
            batch_items = indexed[batch_start:batch_start+BS]
            max_len = max(s['full_len'] for _, s in batch_items)
            B = len(batch_items)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            for i, (_, s) in enumerate(batch_items):
                input_ids[i, :s['full_len']] = torch.tensor(s['full'], device=device)
            attn = (input_ids != pad_id).long()
            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, L, V] bf16
            # Use F.cross_entropy: memory-efficient, no full log_softmax materialization
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            nll_per_pos = torch.nn.functional.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_targets.reshape(-1),
                reduction='none',
            ).reshape(shifted_targets.shape)  # [B, L-1]
            del logits, shifted_logits, shifted_targets  # free intermediate before next iter
            for i, (orig_idx, s) in enumerate(batch_items):
                start = s['prompt_len'] - 1
                end = s['full_len'] - 1
                if start < 0: start = 0
                if end <= start: continue
                seq_nll = nll_per_pos[i, start:end].sum().item()
                n_tok = end - start
                per_q_records[orig_idx] = {
                    'q_idx': orig_idx,
                    'nll_sum': seq_nll, 'n_tok': n_tok,
                    'ppl': math.exp(seq_nll / n_tok) if n_tok else float('inf'),
                }
                sum_nll += seq_nll; sum_tok += n_tok
            processed += B
            del nll_per_pos
            if (batch_start // BS) % 20 == 0:
                elapsed = time.time() - t0
                pct = processed / len(seqs) * 100
                print(f'  [{target_name}] {processed}/{len(seqs)} ({pct:.0f}%)  '
                      f'cum_NLL/tok={sum_nll/sum_tok:.4f}  PPL={math.exp(sum_nll/sum_tok):.3f}  '
                      f'elapsed={elapsed:.0f}s  bs={BS} max_len={max_len}', flush=True)
                torch.cuda.empty_cache()
    per_q = [per_q_records[i] for i in sorted(per_q_records.keys())]
    mean_nll = sum_nll / sum_tok
    out = {
        'target': target_name,
        'n_q': len(per_q),
        'mean_nll_per_token': mean_nll,
        'mean_ppl': math.exp(mean_nll),
        'total_nll': sum_nll, 'total_tokens': sum_tok,
        'elapsed_s': time.time() - t0,
        'per_q': per_q,
    }
    return out


# Phase 1: base
print('\n=== phase 1/3: base IT ===')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='eager'
)
out_base = compute_ppl(base_model, 'base')
json.dump(out_base, open(OUT_DIR / 'base.json', 'w'))
print(f'  saved → {OUT_DIR / "base.json"}  PPL={out_base["mean_ppl"]:.3f}')

# Phase 2: R15 ck-15
print('\n=== phase 2/3: R15 ck-15 ===')
r15 = PeftModel.from_pretrained(base_model, str(TARGETS['r15_ck15']))
r15 = r15.merge_and_unload()
out_r15 = compute_ppl(r15, 'r15_ck15')
json.dump(out_r15, open(OUT_DIR / 'r15_ck15.json', 'w'))
print(f'  saved → {OUT_DIR / "r15_ck15.json"}  PPL={out_r15["mean_ppl"]:.3f}')
del r15, base_model
torch.cuda.empty_cache()

# Phase 3: R16 step_42 (fresh base load to ensure clean adapter merge)
print('\n=== phase 3/3: R16 step_42 ===')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, attn_implementation='eager'
)
r16 = PeftModel.from_pretrained(base_model, str(TARGETS['r16_step42']))
r16 = r16.merge_and_unload()
out_r16 = compute_ppl(r16, 'r16_step42')
json.dump(out_r16, open(OUT_DIR / 'r16_step42.json', 'w'))
print(f'  saved → {OUT_DIR / "r16_step42.json"}  PPL={out_r16["mean_ppl"]:.3f}')

# Summary
print('\n=== Summary ===')
for tag, o in [('base', out_base), ('r15_ck15', out_r15), ('r16_step42', out_r16)]:
    print(f'  {tag:<12} mean_NLL/tok={o["mean_nll_per_token"]:.4f}  PPL={o["mean_ppl"]:.3f}  '
          f'tokens={o["total_tokens"]}  time={o["elapsed_s"]:.0f}s')
print(f'\nΔPPL: r15-base={out_r15["mean_ppl"]-out_base["mean_ppl"]:+.3f}  '
      f'r16-base={out_r16["mean_ppl"]-out_base["mean_ppl"]:+.3f}')
