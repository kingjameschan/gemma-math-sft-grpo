"""Distill (OMI2 Llama-405B → Gemma2-2B) vs base — combined comparison plot.
   Rows: L1.1 pass@K+maj@K (GSM8K | MATH500aug), L6.1 per-q pass-rate hist, L6.2 10x10 transition.
   All from K=128 verbose (base + distill, both sets), DSmath maj@k (permutation).
"""
import json, math, random
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path('/mnt/d/fine-tuning')
OUT = ROOT / 'v3/E6_distill/outputs'
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

def load(path):
    d = json.load(open(path))['samples']
    golds = [normalize(s.get('gold')) for s in d]
    ans = [[normalize(a) for a in s['any_preds']] for s in d]
    c = [sum(1 for a in row if a is not None and a == g) for row, g in zip(ans, golds)]
    return ans, golds, c

def curves(c_arr, ans, golds, K):
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

# Load all
print('[load] base + distill K=128 both sets', flush=True)
bg_ans, bg_g, bg_c = load(ROOT/'v3/E5_grpo/outputs/k128_merged/base_k128_gsm8k_verbose.json')
dg_ans, dg_g, dg_c = load(OUT/'distill_gsm8k_k128.json')
bm_ans, bm_g, bm_c = load(ROOT/'v3/E5_grpo/outputs/k128_merged_math500_aug_slice/base_k128_math500_aug_verbose.json')
dm_ans, dm_g, dm_c = load(OUT/'distill_math500aug_k128.json')
# align math by question (base superset vs distill 500)
dm_d = json.load(open(OUT/'distill_math500aug_k128.json'))['samples']
bm_d = json.load(open(ROOT/'v3/E5_grpo/outputs/k128_merged_math500_aug_slice/base_k128_math500_aug_verbose.json'))['samples']
bm_by_q = {s['question']: s for s in bm_d}
bm_ans2, bm_g2, bm_c2 = [], [], []
dm_ans2, dm_g2, dm_c2 = [], [], []
for s in dm_d:
    q = s['question']
    if q not in bm_by_q: continue
    bs = bm_by_q[q]; g = normalize(bs.get('gold'))
    ba = [normalize(a) for a in bs['any_preds']]; da = [normalize(a) for a in s['any_preds']]
    bm_ans2.append(ba); bm_g2.append(g); bm_c2.append(sum(1 for a in ba if a==g))
    dm_ans2.append(da); dm_g2.append(g); dm_c2.append(sum(1 for a in da if a==g))

print('[curves] GSM8K...', flush=True)
bgp, bgm = curves(bg_c, bg_ans, bg_g, 128)
dgp, dgm = curves(dg_c, dg_ans, dg_g, 128)
print('[curves] MATH...', flush=True)
bmp, bmm = curves(bm_c2, bm_ans2, bm_g2, 128)
dmp, dmm = curves(dm_c2, dm_ans2, dm_g2, 128)

COL_BASE='black'; COL_D='#c0392b'
fig, axes = plt.subplots(3, 2, figsize=(14, 14), facecolor='white')

def passk_panel(ax, bp, bm_, dp, dm_, title, n):
    ax.plot(KS, [bp[k] for k in KS], '-o', color=COL_BASE, lw=2, ms=4, label='base pass@K')
    ax.plot(KS, [bm_[k] for k in KS], '--', color=COL_BASE, lw=1.5, alpha=.7, label='base maj@K')
    ax.plot(KS, [dp[k] for k in KS], '-o', color=COL_D, lw=2, ms=4, label='distill pass@K')
    ax.plot(KS, [dm_[k] for k in KS], '--', color=COL_D, lw=1.5, alpha=.85, label='distill maj@K')
    ax.set_xscale('log', base=2); ax.set_xticks(KS); ax.set_xticklabels(KS, fontsize=7)
    ax.set_xlabel('K'); ax.set_ylabel('accuracy (%)')
    ax.set_title(f'{title} (n={n})\nbase p@1={bp[1]:.1f} p@128={bp[128]:.1f} | distill p@1={dp[1]:.1f} p@128={dp[128]:.1f}',
                 loc='left', fontsize=9, fontweight='semibold')
    ax.legend(fontsize=7.5, loc='lower right'); ax.grid(alpha=.3)

passk_panel(axes[0,0], bgp, bgm, dgp, dgm, 'L1.1a GSM8K pass@K/maj@K', len(bg_c))
passk_panel(axes[0,1], bmp, bmm, dmp, dmm, 'L1.1b MATH500-aug pass@K/maj@K', len(bm_c2))

def hist_panel(ax, bc, dc, K, title):
    br = np.array(bc)/K; dr = np.array(dc)/K
    bins = np.linspace(0,1,21)
    ax.hist(br, bins=bins, alpha=.5, color=COL_BASE, label=f'base mean={br.mean():.3f}', edgecolor='white', lw=.3)
    ax.hist(dr, bins=bins, alpha=.5, color=COL_D, label=f'distill mean={dr.mean():.3f}', edgecolor='white', lw=.3)
    ax.set_xlabel('per-Q pass-rate (c/K)'); ax.set_ylabel('# questions')
    ax.set_title(f'{title}  Δmean={dr.mean()-br.mean():+.3f}', loc='left', fontsize=9, fontweight='semibold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

hist_panel(axes[1,0], bg_c, dg_c, 128, 'L6.1a GSM8K per-q pass-rate')
hist_panel(axes[1,1], bm_c2, dm_c2, 128, 'L6.1b MATH per-q pass-rate')

def trans_panel(ax, bc, dc, K, title):
    def b10(r): return min(int(r/K*10), 9)
    M = np.zeros((10,10), int)
    for b, d in zip(bc, dc): M[b10(b), b10(d)] += 1
    im = ax.imshow(M, cmap='Blues', norm=LogNorm(vmin=1, vmax=max(M.max(),1)), origin='lower', aspect='equal')
    for i in range(10):
        for j in range(10):
            if M[i,j] >= 3: ax.text(j,i,M[i,j],ha='center',va='center',fontsize=6,color='white' if M[i,j]>M.max()/3 else 'black')
    ax.plot([-.5,9.5],[-.5,9.5],'r--',lw=.6,alpha=.5)
    ax.set_xlabel('distill pass-rate decile'); ax.set_ylabel('base pass-rate decile')
    ax.set_title(title, loc='left', fontsize=9, fontweight='semibold')
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)

trans_panel(axes[2,0], bg_c, dg_c, 128, 'L6.2a GSM8K base→distill transition')
trans_panel(axes[2,1], bm_c2, dm_c2, 128, 'L6.2b MATH base→distill transition')

fig.suptitle('E6 Distill (OpenMathInstruct-2 / Llama-3.1-405B CoT, 100K×2ep) vs base Gemma2-2B-IT — K=128',
             fontsize=12, fontweight='semibold', y=1.0)
plt.tight_layout()
outp = OUT / 'distill_combined.png'
plt.savefig(outp, dpi=180, bbox_inches='tight', facecolor='white')
print(f'saved {outp}', flush=True)
