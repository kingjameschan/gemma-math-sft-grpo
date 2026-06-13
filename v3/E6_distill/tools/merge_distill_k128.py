"""Merge distill seed42 + seed43 K=64 eval → K=128 verbose, per test set.
   Align by question, concat any_preds, recompute any_correct via local normalize.
   Output: v3/E6_distill/outputs/distill_<set>_k128.json
"""
import json, math, glob
from collections import Counter
from pathlib import Path

ROOT = Path('/mnt/d/fine-tuning')
OUT = ROOT / 'v3/E6_distill/outputs'

def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError, OverflowError):
        return s

def pak(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

for setname in ['gsm8k', 'math500aug']:
    f42 = glob.glob(str(OUT / f'pass_at_k_distill_{setname}_s42_*/*_k64.json'))[0]
    f43 = glob.glob(str(OUT / f'pass_at_k_distill_{setname}_s43_*/*_k64.json'))[0]
    d42 = json.load(open(f42)); d43 = json.load(open(f43))
    by_q = {s['question']: s for s in d42['samples']}
    merged = []
    for s in d43['samples']:
        q = s['question']
        if q not in by_q: continue
        a = by_q[q]
        g = a.get('gold')
        any_preds = a.get('any_preds', []) + s.get('any_preds', [])
        boxed_preds = a.get('boxed_preds', []) + s.get('boxed_preds', [])
        responses = a.get('responses', []) + s.get('responses', [])
        gn = normalize(g)
        any_correct = sum(1 for p in any_preds if p is not None and normalize(p) == gn)
        boxed_correct = sum(1 for p in boxed_preds if p is not None and normalize(p) == gn)
        valid = [normalize(p) for p in any_preds if p]
        maj = Counter(valid).most_common(1)[0][0] if valid else None
        merged.append({
            'question': q, 'gold': g,
            'any_correct_per_K': any_correct,
            'boxed_correct_per_K': boxed_correct,
            'any_preds': any_preds, 'boxed_preds': boxed_preds,
            'responses': responses,
            'maj_numeric_pred': maj,
        })
    n = len(merged)
    K = 128
    metrics = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        metrics[f'pass_at_{k}_numeric'] = sum(pak(s['any_correct_per_K'], K, k) for s in merged) / n
    out = {'config': {'K': 128, 'samples': n, 'source': 'merge s42+s43', 'task': setname},
           'metrics': metrics, 'samples': merged}
    outf = OUT / f'distill_{setname}_k128.json'
    json.dump(out, open(outf, 'w'))
    print(f'{setname}: N={n} K=128  p@1={metrics["pass_at_1_numeric"]:.4f}  p@64={metrics["pass_at_64_numeric"]:.4f}  p@128={metrics["pass_at_128_numeric"]:.4f}')
