"""Merge E7 pretrain-base eval seed42+43 K=64 → K=128, for BOTH ptbase (before) and
   ptsft (after), both test sets. Output: v3/E6_distill/outputs/distill_pt_<tag>_k128.json
   tag in {ptbase_gsm8k, ptsft_gsm8k, ptbase_math500aug, ptsft_math500aug}.
"""
import json, math, glob
from collections import Counter
from pathlib import Path

OUT = Path('/mnt/d/fine-tuning/v3/E6_distill/outputs')

def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        v = float(s); return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s

def pak(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

for tag in ['ptbase_gsm8k', 'ptsft_gsm8k', 'ptbase_math500aug', 'ptsft_math500aug']:
    g42 = glob.glob(str(OUT / f'pass_at_k_{tag}_s42_*/*_k64.json'))
    g43 = glob.glob(str(OUT / f'pass_at_k_{tag}_s43_*/*_k64.json'))
    if not g42 or not g43:
        print(f'{tag}: MISSING (s42={len(g42)} s43={len(g43)}) — skip'); continue
    d42 = json.load(open(g42[0])); d43 = json.load(open(g43[0]))
    by_q = {s['question']: s for s in d42['samples']}
    merged = []
    for s in d43['samples']:
        q = s['question']
        if q not in by_q: continue
        a = by_q[q]; g = a.get('gold'); gn = normalize(g)
        any_preds = a.get('any_preds', []) + s.get('any_preds', [])
        any_correct = sum(1 for p in any_preds if p is not None and normalize(p) == gn)
        valid = [normalize(p) for p in any_preds if p]
        maj = Counter(valid).most_common(1)[0][0] if valid else None
        merged.append({'question': q, 'gold': g, 'any_correct_per_K': any_correct,
                       'any_preds': any_preds, 'maj_numeric_pred': maj})
    n = len(merged); K = 128
    metrics = {f'pass_at_{k}_numeric': sum(pak(s['any_correct_per_K'], K, k) for s in merged) / n
               for k in [1, 2, 4, 8, 16, 32, 64, 128]}
    out = {'config': {'K': 128, 'samples': n, 'tag': tag}, 'metrics': metrics, 'samples': merged}
    json.dump(out, open(OUT / f'distill_pt_{tag}_k128.json', 'w'))
    print(f'{tag}: N={n} p@1={metrics["pass_at_1_numeric"]:.4f} p@64={metrics["pass_at_64_numeric"]:.4f} p@128={metrics["pass_at_128_numeric"]:.4f}')
