"""E7 before/after combined plot: pretrain gemma-2-2b BASE (before) vs
   pretrain+distill-SFT (after), K=128, GSM8K + math500aug.
   Rows: L1.1 pass@K+maj@K, L6.1 per-q pass-rate hist, L6.2 10x10 transition.
"""
import json, math, random
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = Path('/mnt/d/fine-tuning/v3/E6_distill/outputs')
KS = [1, 2, 4, 8, 16, 32, 64, 128]

def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        v = float(s); return str(int(v)) if v == int(v) else str(v)
    except Exception: return s

def pak(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def load(tag):
    d = json.load(open(OUT / f'distill_pt_{tag}_k128.json'))['samples']
    golds = [normalize(s.get('gold')) for s in d]
    ans = [[normalize(a) for a in s['any_preds']] for s in d]
    c = [s['any_correct_per_K'] for s in d]
    return ans, golds, c, {s['question']: s for s in d}

def curves(c_arr, ans, golds, K=128):
    rng = random.Random(42); T = 200
    passk, majk = {}, {}
    for k in KS:
        passk[k] = sum(pak(c, K, k) for c in c_arr) / len(c_arr) * 100
        ms = 0.0
        for row, g in zip(ans, golds):
            if not row: continue
            h = 0
            for _ in range(T):
                sh = rng.sample(row, min(k, len(row)))
                v = [a for a in sh if a is not None]
                if v and Counter(v).most_common(1)[0][0] == g: h += 1
            ms += h / T
        majk[k] = ms / len(ans) * 100
    return passk, majk

def align(before_d, after_ans, after_golds, after_c, after_q):
    # align by question (both eval'd same test set, same order expected, but be safe)
    bd_by_q = before_d
    ba, bg, bc, aa, ag, ac = [], [], [], [], [], []
    for q, s in after_q.items():
        if q not in bd_by_q: continue
        bs = bd_by_q[q]
        ba.append([normalize(x) for x in bs['any_preds']]); bg.append(normalize(bs.get('gold'))); bc.append(bs['any_correct_per_K'])
        aa.append([normalize(x) for x in s['any_preds']]); ag.append(normalize(s.get('gold'))); ac.append(s['any_correct_per_K'])
    return (ba, bg, bc), (aa, ag, ac)

COL_B = 'black'; COL_A = '#1f8a4c'  # before black, after green
fig, axes = plt.subplots(3, 2, figsize=(14, 14), facecolor='white')

for col, setname, label in [(0, 'gsm8k', 'GSM8K'), (1, 'math500aug', 'MATH500-aug')]:
    ba, bg, bc, bq = load(f'ptbase_{setname}')
    aa, ag, ac, aq = load(f'ptsft_{setname}')
    (b_ans, b_g, b_c), (a_ans, a_g, a_c) = align(bq, aa, ag, ac, aq)
    n = len(b_c)
    bp, bm = curves(b_c, b_ans, b_g)
    ap_, am = curves(a_c, a_ans, a_g)
    # row0 pass@K/maj@K
    ax = axes[0, col]
    ax.plot(KS, [bp[k] for k in KS], '-o', color=COL_B, lw=2, ms=4, label='before(base) pass@K')
    ax.plot(KS, [bm[k] for k in KS], '--', color=COL_B, lw=1.4, alpha=.7, label='before maj@K')
    ax.plot(KS, [ap_[k] for k in KS], '-o', color=COL_A, lw=2, ms=4, label='after(distill) pass@K')
    ax.plot(KS, [am[k] for k in KS], '--', color=COL_A, lw=1.4, alpha=.85, label='after maj@K')
    ax.set_xscale('log', base=2); ax.set_xticks(KS); ax.set_xticklabels(KS, fontsize=7)
    ax.set_xlabel('K'); ax.set_ylabel('accuracy (%)')
    ax.set_title(f'L1.1 {label} (n={n})\nbefore p@1={bp[1]:.1f} p@128={bp[128]:.1f} | after p@1={ap_[1]:.1f} p@128={ap_[128]:.1f}',
                 loc='left', fontsize=9, fontweight='semibold')
    ax.legend(fontsize=7.5, loc='lower right'); ax.grid(alpha=.3)
    # row1 per-q pass-rate hist
    ax = axes[1, col]
    br = np.array(b_c)/128; ar = np.array(a_c)/128
    bins = np.linspace(0, 1, 21)
    ax.hist(br, bins=bins, alpha=.5, color=COL_B, label=f'before mean={br.mean():.3f}', edgecolor='white', lw=.3)
    ax.hist(ar, bins=bins, alpha=.5, color=COL_A, label=f'after mean={ar.mean():.3f}', edgecolor='white', lw=.3)
    ax.set_xlabel('per-Q pass-rate (c/K)'); ax.set_ylabel('# questions')
    ax.set_title(f'L6.1 {label} per-q pass-rate  Δmean={ar.mean()-br.mean():+.3f}', loc='left', fontsize=9, fontweight='semibold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
    # row2 transition
    ax = axes[2, col]
    def b10(r): return min(int(r/128*10), 9)
    M = np.zeros((10, 10), int)
    for b, a in zip(b_c, a_c): M[b10(b), b10(a)] += 1
    im = ax.imshow(M, cmap='Greens', norm=LogNorm(vmin=1, vmax=max(M.max(), 1)), origin='lower', aspect='equal')
    for i in range(10):
        for j in range(10):
            if M[i, j] >= 3: ax.text(j, i, M[i, j], ha='center', va='center', fontsize=6, color='white' if M[i, j] > M.max()/3 else 'black')
    ax.plot([-.5, 9.5], [-.5, 9.5], 'r--', lw=.6, alpha=.5)
    ax.set_xlabel('after pass-rate decile'); ax.set_ylabel('before pass-rate decile')
    ax.set_title(f'L6.2 {label} before→after transition', loc='left', fontsize=9, fontweight='semibold')
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)

fig.suptitle('E7 Distill on PRETRAIN gemma-2-2b (non-it): before(base) vs after(distill-SFT 100K×2ep), K=128',
             fontsize=12, fontweight='semibold', y=1.0)
plt.tight_layout()
outp = OUT / 'distill_pretrain_combined.png'
plt.savefig(outp, dpi=180, bbox_inches='tight', facecolor='white')
print(f'saved {outp}')
