"""Reusable paper-style panels:
- paired histogram of per-question pass-rate (10-bin), base vs ckpt
- 10x10 transition heatmap (base bin -> ckpt bin)

Usage:
    from _paper_style_panels import compute_pass_rates, plot_paired_hist, plot_transition_10x10
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def compute_pass_rates_verbose(file_path):
    """Verbose schema: {samples: [{any_correct_per_K, ...}], config: {K}}"""
    d = json.load(open(file_path))
    K = d['config']['K']
    rates = [s['any_correct_per_K'] / K for s in d['samples']]
    return np.array(rates), K, len(rates)


def compute_pass_rates_slim(file_path):
    """Slim schema: {per_q: [{gold, samples: [{c, a, L}]}], K, n}"""
    d = json.load(open(file_path))
    K = d.get('K') or len(d['per_q'][0]['samples'])
    rates = [sum(s['c'] for s in q['samples']) / K for q in d['per_q']]
    return np.array(rates), K, len(rates)


def compute_pass_rates(file_path):
    """Auto-detect schema."""
    d = json.load(open(file_path))
    if 'per_q' in d:
        return compute_pass_rates_slim(file_path)
    elif 'samples' in d and isinstance(d['samples'], list) and 'any_correct_per_K' in d['samples'][0]:
        return compute_pass_rates_verbose(file_path)
    else:
        raise ValueError(f"unknown schema: {list(d.keys())}")


BIN12_LABELS = ['=0'] + [f'(.{i},.{i+1}]' for i in range(0, 9)] + ['(.9,1)', '=1']


def plot_l2_perbin_pass1(ax, base_rates, ckpt_rates, ckpt_label='ckpt', K=64, title_prefix='L2.1'):
    """Per-bin pass@1 grouped bar chart, 12-bin by base difficulty.
    For each base bin, plot mean pass@1 (base) vs mean pass@1 (ckpt).
    """
    base_idx = _bin12(base_rates)
    by_bin_base = np.zeros(12); by_bin_ckpt = np.zeros(12); sizes = np.zeros(12, dtype=int)
    for i in range(12):
        mask = base_idx == i
        sizes[i] = mask.sum()
        if sizes[i] == 0: continue
        by_bin_base[i] = base_rates[mask].mean() * 100
        by_bin_ckpt[i] = ckpt_rates[mask].mean() * 100
    x = np.arange(12); w = 0.4
    ax.bar(x - w/2, by_bin_base, w, color='#888', label='base IT', alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.bar(x + w/2, by_bin_ckpt, w, color='#7c3aed', label=ckpt_label, alpha=0.85, edgecolor='black', linewidth=0.3)
    for i in range(12):
        if sizes[i] > 0:
            d = by_bin_ckpt[i] - by_bin_base[i]
            color = '#16a34a' if d >= 0 else '#dc2626'
            ax.text(i, max(by_bin_base[i], by_bin_ckpt[i]) + 4, f'{d:+.1f}', ha='center',
                    fontsize=6.5, color=color, fontweight='semibold')
            ax.text(i, -7, f'n={sizes[i]}', ha='center', fontsize=6, color='gray')
    ax.set_xticks(x); ax.set_xticklabels(BIN12_LABELS, fontsize=6.5, rotation=45, ha='right')
    ax.set_xlabel('base pass-rate bin (difficulty)')
    ax.set_ylabel('mean pass@1 (%)')
    ax.set_ylim(-12, 115)
    for x_special, color in [(0, '#fee2e2'), (11, '#dcfce7')]:
        ax.axvspan(x_special - 0.5, x_special + 0.5, alpha=0.5, color=color, zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), fontsize=8, ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{title_prefix} — per-bin pass@1 (12-bin by base difficulty, K={K})',
                 loc='left', fontsize=9, fontweight='semibold')


def plot_l2_size_stack(ax, base_rates, ckpt_rates, ckpt_label='ckpt', title_prefix='L2.2.left'):
    """Side-by-side bar showing # questions per 12 bin (base vs ckpt distribution).
    Like L6.1 but using consistent 12-bin scheme and showing absolute counts.
    """
    base_idx = _bin12(base_rates)
    ckpt_idx = _bin12(ckpt_rates)
    h_b = np.array([(base_idx == i).sum() for i in range(12)])
    h_c = np.array([(ckpt_idx == i).sum() for i in range(12)])
    x = np.arange(12); w = 0.4
    ax.bar(x - w/2, h_b, w, color='#888', label='base IT', edgecolor='black', linewidth=0.3)
    ax.bar(x + w/2, h_c, w, color='#7c3aed', label=ckpt_label, edgecolor='black', linewidth=0.3)
    for i in range(12):
        if h_b[i] >= 10: ax.text(i - w/2, h_b[i] + 2, str(h_b[i]), ha='center', fontsize=5.5, color='black')
        if h_c[i] >= 10: ax.text(i + w/2, h_c[i] + 2, str(h_c[i]), ha='center', fontsize=5.5, color='#5b21b6')
    ax.set_xticks(x); ax.set_xticklabels(BIN12_LABELS, fontsize=6.5, rotation=45, ha='right')
    ax.set_xlabel('per-q pass-rate bin')
    ax.set_ylabel('# questions')
    for x_special, color in [(0, '#fee2e2'), (11, '#dcfce7')]:
        ax.axvspan(x_special - 0.5, x_special + 0.5, alpha=0.5, color=color, zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), fontsize=8, ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{title_prefix} — bin size dist (12-bin)',
                 loc='left', fontsize=9, fontweight='semibold')


def _bin12(rates):
    """12-bin: idx 0 = exact 0, idx 11 = exact 1.0, idx 1..10 = (0, 0.1] (0.1, 0.2] ... (0.9, 1.0)."""
    rates = np.asarray(rates)
    idx = np.full(len(rates), -1, dtype=int)
    idx[rates == 0.0] = 0
    idx[rates == 1.0] = 11
    mid = (rates > 0) & (rates < 1)
    # Bin (0, 0.1] → 1, (0.1, 0.2] → 2, ... (0.9, 1.0) → 10
    idx[mid] = np.clip(np.ceil(rates[mid] * 10).astype(int), 1, 10)
    return idx


def plot_paired_hist(ax, rates_base, rates_ckpt, title=None, labels=('base', 'ckpt')):
    """12-bin paired histogram. idx 0 = pass=0 (all wrong, unsolvable), idx 11 = pass=1 (all right, mastered),
    idx 1..10 = (0, 0.1] (0.1, 0.2] ... (0.9, 1.0) — 10 intermediate stochastic bins.
    """
    idx_b = _bin12(rates_base)
    idx_c = _bin12(rates_ckpt)
    h_base = np.array([(idx_b == i).sum() for i in range(12)])
    h_ckpt = np.array([(idx_c == i).sum() for i in range(12)])

    x = np.arange(12)
    width = 0.4
    ax.bar(x - width/2, h_base, width=width, color='#888', label=labels[0], edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, h_ckpt, width=width, color='#2c7fb8', label=labels[1], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('per-question pass-rate (c/K)')
    ax.set_ylabel('# questions')
    ax.set_xticks(range(12))
    ax.set_xticklabels(BIN12_LABELS, fontsize=6.5, rotation=45, ha='right')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    # Highlight 0 and 1 bars with bold edge
    for i_special, x_special in [(0, 0), (11, 11)]:
        ax.axvspan(x_special - 0.5, x_special + 0.5, alpha=0.08, color='red' if i_special == 0 else 'green', zorder=0)
    if title:
        ax.set_title(title, fontsize=10)


def plot_transition_10x10(ax, rates_base, rates_ckpt, title=None, log_scale=True):
    """12x12 transition heatmap (base bin row -> ckpt bin col).

    idx 0 = pass=0 (none correct), idx 11 = pass=1 (all correct).
    Diagonal = no change. Above diag (j>i) = improvement.
    Log-scale color so sparse mid-cells visible vs huge corners.
    """
    base_idx = _bin12(rates_base)
    ckpt_idx = _bin12(rates_ckpt)
    n_bin = 12
    M = np.zeros((n_bin, n_bin), dtype=int)
    for b, c in zip(base_idx, ckpt_idx):
        M[b, c] += 1

    if log_scale:
        cmap_norm = LogNorm(vmin=max(1, M[M > 0].min() if (M > 0).any() else 1), vmax=max(M.max(), 1))
    else:
        cmap_norm = None
    im = ax.imshow(M, cmap='Blues', norm=cmap_norm, aspect='equal', origin='lower')

    for i in range(n_bin):
        for j in range(n_bin):
            v = M[i, j]
            if v >= 5:
                ax.text(j, i, str(v), ha='center', va='center',
                        color='white' if v > M.max() / 3 else 'black', fontsize=6)

    # Diagonal line
    ax.plot([-0.5, n_bin - 0.5], [-0.5, n_bin - 0.5], color='red', linewidth=0.6, linestyle='--', alpha=0.5)

    ax.set_xticks(range(n_bin)); ax.set_xticklabels(BIN12_LABELS, fontsize=5.5, rotation=45, ha='right')
    ax.set_yticks(range(n_bin)); ax.set_yticklabels(BIN12_LABELS, fontsize=5.5)
    ax.set_xlabel('ckpt pass-rate bin', fontsize=8)
    ax.set_ylabel('base pass-rate bin', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # No text overlay — keep heatmap clean per user feedback


def plot_abc5_bar(ax, base_c, base_mc, base_mm, ckpt_c, ckpt_mc, ckpt_mm,
                   ckpt_label, dataset_name, title_prefix='L1.2'):
    """5-bucket grouped bar: A_sharp / A_diffuse / B_sharp / B_diffuse / C.
    mm = mode mass = top freq / K."""
    def split5(c_arr, mc_arr, mm_arr):
        n = len(c_arr)
        out = {"A_sharp": 0, "A_diffuse": 0, "B_sharp": 0, "B_diffuse": 0, "C": 0}
        for c, mc, mm in zip(c_arr, mc_arr, mm_arr):
            if c == 0: out["C"] += 1
            elif mc:
                if mm > 0.5: out["A_sharp"] += 1
                else: out["A_diffuse"] += 1
            else:
                if mm > 0.5: out["B_sharp"] += 1
                else: out["B_diffuse"] += 1
        return {k: v / n * 100 for k, v in out.items()}
    bb = split5(base_c, base_mc, base_mm)
    sb = split5(ckpt_c, ckpt_mc, ckpt_mm)
    bucket_order = ["A_sharp", "A_diffuse", "B_sharp", "B_diffuse", "C"]
    bucket_label = ["A_sharp\n(mode ok\nmm>0.5)", "A_diffuse\n(mode ok\nmm≤0.5)",
                    "B_sharp\n(mode wrong\nmm>0.5)", "B_diffuse\n(mode wrong\nmm≤0.5)",
                    "C\n(c=0)"]
    base_color = ["#86efac", "#bbf7d0", "#fca5a5", "#fed7aa", "#9ca3af"]
    ckpt_color = ["#16a34a", "#65a30d", "#dc2626", "#f59e0b", "#525252"]
    x = np.arange(len(bucket_order)); w = 0.38
    base_vals = [bb[k] for k in bucket_order]
    ckpt_vals = [sb[k] for k in bucket_order]
    ax.bar(x - w/2, base_vals, w, color=base_color, edgecolor="black", linewidth=0.5, label="base")
    ax.bar(x + w/2, ckpt_vals, w, color=ckpt_color, edgecolor="black", linewidth=0.5, label=ckpt_label)
    for i, (b, s) in enumerate(zip(base_vals, ckpt_vals)):
        ax.text(i - w/2, b + 0.8, f"{b:.1f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, s + 0.8, f"{s:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(bucket_label, fontsize=7.5)
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, max(max(base_vals), max(ckpt_vals)) * 1.22)
    ax.set_title(f"{title_prefix} [{dataset_name}] — 5-bucket distribution (mm = mode mass)",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper right", fontsize=8); ax.grid(axis="y", alpha=0.25, linestyle=":")


def plot_abc5_transition(ax, base_c, base_mc, base_mm, ckpt_c, ckpt_mc, ckpt_mm,
                          ckpt_label, dataset_name, title_prefix='L1.2.right'):
    """5x5 transition matrix on 5-bucket index: 0=A_sharp 1=A_diffuse 2=B_sharp 3=B_diffuse 4=C."""
    def _bk5(c, mc, mm):
        if c == 0: return 4
        if mc: return 0 if mm > 0.5 else 1
        return 2 if mm > 0.5 else 3
    M = np.zeros((5, 5), dtype=int)
    for c_b, mc_b, mm_b, c_d, mc_d, mm_d in zip(base_c, base_mc, base_mm,
                                                  ckpt_c, ckpt_mc, ckpt_mm):
        M[_bk5(int(c_b), int(mc_b), float(mm_b)), _bk5(int(c_d), int(mc_d), float(mm_d))] += 1
    ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(5):
        row_total = M[i].sum()
        for j in range(5):
            cnt = M[i, j]; pct = cnt / row_total * 100 if row_total else 0
            color = "white" if cnt > M.max() * 0.5 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=7, fontweight="semibold")
    labels = ["A_sharp", "A_diffuse", "B_sharp", "B_diffuse", "C"]
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"→ {ckpt_label}", fontsize=8.5); ax.set_ylabel("base", fontsize=8.5)
    ax.set_title(f"{title_prefix} [{dataset_name}] — 5×5 transition matrix",
                 loc="left", fontsize=9, fontweight="semibold")


def plot_abc_bar(ax, base_c, base_mc, ckpt_c, ckpt_mc, ckpt_label, dataset_name,
                  color_ckpt='#3498db', title_prefix='L1.2'):
    """ABC 3-bucket grouped bar: A=mode correct, B=≥1 correct & mode wrong, C=0 correct."""
    def abc(c_arr, mc_arr):
        n = len(c_arr)
        not_lost = sum(1 for c in c_arr if c >= 1) / n * 100
        A = sum(1 for mc in mc_arr if mc) / n * 100
        B = not_lost - A; C = 100 - not_lost
        return A, B, C
    A_b, B_b, C_b = abc(base_c, base_mc)
    A_s, B_s, C_s = abc(ckpt_c, ckpt_mc)
    base_vals = [A_b, B_b, C_b]; sft_vals = [A_s, B_s, C_s]
    x = np.arange(3); w = 0.4
    ax.bar(x - w/2, base_vals, w, color='#888', label='base IT', alpha=0.85,
           edgecolor='black', linewidth=0.3)
    ax.bar(x + w/2, sft_vals, w, color=color_ckpt, label=ckpt_label, alpha=0.85,
           edgecolor='black', linewidth=0.3)
    for i, (bv, sv) in enumerate(zip(base_vals, sft_vals)):
        ax.text(i - w/2, bv + 1.5, f"{bv:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, sv + 1.5, f"{sv:.1f}", ha="center", fontsize=8, color=color_ckpt)
    ax.set_xticks(x)
    ax.set_xticklabels(["A\n(mode correct)", "B\n(≥1 correct,\nmode wrong)", "C\n(0 correct)"], fontsize=8.5)
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, max(max(base_vals), max(sft_vals)) * 1.2)
    ax.set_title(f"{title_prefix} [{dataset_name}] — ABC 3-bucket distribution",
                 loc="left", fontsize=9, fontweight="semibold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)


def plot_abc_transition(ax, base_c, base_mc, ckpt_c, ckpt_mc, ckpt_label, dataset_name,
                         title_prefix='L1.2.right'):
    M = np.zeros((3, 3), dtype=int)
    def _bk(c, mc):
        if c == 0: return 2
        if mc: return 0
        return 1
    for c_b, mc_b, c_d, mc_d in zip(base_c, base_mc, ckpt_c, ckpt_mc):
        M[_bk(int(c_b), int(mc_b)), _bk(int(c_d), int(mc_d))] += 1
    ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(3):
        row_total = M[i].sum()
        for j in range(3):
            cnt = M[i, j]; pct = cnt / row_total * 100 if row_total else 0
            color = "white" if cnt > M.max() * 0.5 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.2f}%)", ha="center", va="center",
                    color=color, fontsize=10, fontweight="semibold")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["A", "B", "C"], fontsize=9)
    ax.set_yticklabels(["A", "B", "C"], fontsize=9)
    ax.set_xlabel(f"→ {ckpt_label}", fontsize=8.5); ax.set_ylabel("base", fontsize=8.5)
    ax.set_title(f"{title_prefix} [{dataset_name}] — A/B/C transition matrix",
                 loc="left", fontsize=9, fontweight="semibold")


def plot_l7_delta_mode_mass(ax, base_ans, ckpt_ans, golds, ckpt_label='ckpt',
                             title_prefix='L7'):
    """Per-question Δmode_mass = mode_mass_ckpt - mode_mass_base, stacked by ckpt ABC bucket.
    base_ans, ckpt_ans: list of list of normalized answer strings (1 per chain).
    golds: list of gold answer strings (1 per question).
    """
    from collections import Counter
    def mode_mass(preds):
        valid = [p for p in preds if p is not None and p != '']
        if not valid: return 0.0, None
        ct = Counter(valid)
        a, c = ct.most_common(1)[0]
        return c / len(preds), a

    deltas = []; buckets = []
    for bp, cp, g in zip(base_ans, ckpt_ans, golds):
        mm_b, _ = mode_mass(bp[:len(cp) if len(cp) <= len(bp) else 64])
        mm_c, mode_a = mode_mass(cp)
        d = mm_c - mm_b
        n_corr = sum(1 for p in cp if p is not None and str(p).strip() == str(g).strip())
        if n_corr == 0:
            bucket = 'C'
        elif mode_a is not None and str(mode_a).strip() == str(g).strip():
            bucket = 'A'
        else:
            bucket = 'B'
        deltas.append(d); buckets.append(bucket)
    deltas = np.array(deltas); buckets = np.array(buckets)
    bins = np.arange(-1.0, 1.0001, 0.05)
    colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#95a5a6'}
    labels_full = {'A': 'A: mode=gold (sharpen on correct)',
                   'B': 'B: mode≠gold ≥1 correct (sharpen on wrong)',
                   'C': 'C: 0/K correct'}
    order = ['A', 'B', 'C']
    data_list = [deltas[buckets == b] for b in order]
    ax.hist(data_list, bins=bins, stacked=True,
            color=[colors[b] for b in order],
            label=[f'{labels_full[b]} (n={len(d)})' for b, d in zip(order, data_list)],
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    mu_all = float(deltas.mean())
    mu_A = float(deltas[buckets == 'A'].mean()) if len(data_list[0]) else float('nan')
    mu_B = float(deltas[buckets == 'B'].mean()) if len(data_list[1]) else float('nan')
    ax.axvline(mu_all, color='black', linestyle=':', linewidth=1.2, alpha=0.9, label=f'mean all={mu_all:+.3f}')
    ax.axvline(mu_A, color='#1e8449', linestyle=':', linewidth=1.0, alpha=0.9, label=f'mean A={mu_A:+.3f}')
    ax.axvline(mu_B, color='#a93226', linestyle=':', linewidth=1.0, alpha=0.9, label=f'mean B={mu_B:+.3f}')
    n_pos = int((deltas > 0).sum()); n_neg = int((deltas < 0).sum()); n_zero = int((deltas == 0).sum())
    ax.text(0.02, 0.98, f'#Δ>0: {n_pos}\n#Δ<0: {n_neg}\n#Δ=0: {n_zero}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r'$\Delta$mode_mass = mode_mass$_{ckpt}$ − mode_mass$_{base}$')
    ax.set_ylabel('# questions')
    ax.set_title(f"{title_prefix} — per-q Δmode_mass distribution ({ckpt_label}, stacked A/B/C)",
                 loc='left', fontsize=9, fontweight='semibold')
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, linestyle=':')


def plot_l8_delta_mass_base_anchor(ax, base_ans, ckpt_ans, golds, ckpt_label='ckpt',
                                    title_prefix='L8'):
    """Per-question Δmass anchored on BASE mode:
       Δ = freq_ckpt(Y_base_mode) - freq_base(Y_base_mode)
       — "did RL amplify or suppress base's most-common answer?"
       Buckets:
         A: base mode = gold (base already correct)
         B: base mode ≠ gold, ckpt ≥1 correct (base wrong, recoverable)
         C: ckpt 0/K correct
    """
    from collections import Counter
    def _base_mode(preds):
        valid = [p for p in preds if p is not None and p != '']
        if not valid: return None, 0.0
        a, c = Counter(valid).most_common(1)[0]
        return a, c / len(preds)
    def _freq(preds, target):
        if target is None or not preds: return 0.0
        c = sum(1 for p in preds if p is not None and str(p).strip() == str(target).strip())
        return c / len(preds)

    deltas, buckets = [], []
    for bp, cp, g in zip(base_ans, ckpt_ans, golds):
        bp_use = bp[:len(cp) if len(cp) <= len(bp) else 64]
        Yb, mb = _base_mode(bp_use)
        if Yb is None: continue
        mc = _freq(cp, Yb)
        d = mc - mb
        n_corr = sum(1 for p in cp if p is not None and str(p).strip() == str(g).strip())
        if n_corr == 0: bucket = 'C'
        elif str(Yb).strip() == str(g).strip(): bucket = 'A'
        else: bucket = 'B'
        deltas.append(d); buckets.append(bucket)
    deltas = np.array(deltas); buckets = np.array(buckets)
    bins = np.arange(-1.0, 1.0001, 0.05)
    colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#95a5a6'}
    labels = {'A': 'A: base mode = gold (base already correct)',
              'B': 'B: base mode ≠ gold, ckpt ≥1 correct',
              'C': 'C: ckpt 0/K correct'}
    order = ['A', 'B', 'C']
    data_list = [deltas[buckets == b] for b in order]
    ax.hist(data_list, bins=bins, stacked=True,
            color=[colors[b] for b in order],
            label=[f'{labels[b]} (n={len(d)})' for b, d in zip(order, data_list)],
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    mu_all = float(deltas.mean()) if len(deltas) else float('nan')
    mu_A = float(deltas[buckets == 'A'].mean()) if len(data_list[0]) else float('nan')
    mu_B = float(deltas[buckets == 'B'].mean()) if len(data_list[1]) else float('nan')
    mu_C = float(deltas[buckets == 'C'].mean()) if len(data_list[2]) else float('nan')
    ax.axvline(mu_all, color='black', linestyle=':', linewidth=1.2, alpha=0.9, label=f'mean all={mu_all:+.3f}')
    ax.axvline(mu_A, color='#1e8449', linestyle=':', linewidth=1.0, alpha=0.9, label=f'mean A={mu_A:+.3f}')
    ax.axvline(mu_B, color='#a93226', linestyle=':', linewidth=1.0, alpha=0.9, label=f'mean B={mu_B:+.3f}')
    n_pos = int((deltas > 0).sum()); n_neg = int((deltas < 0).sum()); n_zero = int((deltas == 0).sum())
    sum_pos = float(deltas[deltas > 0].sum()) if n_pos else 0.0
    sum_neg = float(deltas[deltas < 0].sum()) if n_neg else 0.0
    sum_abs = sum_pos - sum_neg  # total |Δ| volume
    net = sum_pos + sum_neg
    ax.text(0.02, 0.98,
            f'#Δ>0: {n_pos}  ΣΔ>0: {sum_pos:+.2f}\n'
            f'#Δ<0: {n_neg}  ΣΔ<0: {sum_neg:+.2f}\n'
            f'#Δ=0: {n_zero}\n'
            f'net ΣΔ: {net:+.2f}\n'
            f'total |Δ|: {sum_abs:.2f}\n'
            f'mean C={mu_C:+.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r'$\Delta$mass($Y_{base\_mode}$) = freq$_{ckpt}(Y_{base\_mode})$ − freq$_{base}(Y_{base\_mode})$')
    ax.set_ylabel('# questions')
    ax.set_title(f"{title_prefix} — Δmass of BASE's mode answer ({ckpt_label}, anchored on Y_base_mode)",
                 loc='left', fontsize=9, fontweight='semibold')
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, linestyle=':')


def _l9_l10_compute(base_ans, ckpt_ans, golds):
    """Shared compute: per-Q deltas + base mode info."""
    from collections import Counter
    def _base_mode(preds):
        valid = [p for p in preds if p is not None and p != '']
        if not valid: return None, 0.0
        a, c = Counter(valid).most_common(1)[0]
        return a, c / len(preds)
    def _freq(preds, target):
        if target is None or not preds: return 0.0
        return sum(1 for p in preds if p is not None and str(p).strip() == str(target).strip()) / len(preds)
    rows = []
    for bp, cp, g in zip(base_ans, ckpt_ans, golds):
        bp_use = bp[:len(cp) if len(cp) <= len(bp) else 64]
        Yb, mb = _base_mode(bp_use)
        if Yb is None: continue
        mc = _freq(cp, Yb)
        base_c = sum(1 for p in bp_use if p is not None and str(p).strip() == str(g).strip())
        rows.append({
            'd': mc - mb,
            'base_mm': mb,
            'base_mode_correct': str(Yb).strip() == str(g).strip(),
            'base_c': base_c,
        })
    return rows


def _stacked_delta_hist(ax, deltas_by_bucket, bucket_order, bucket_colors, bucket_labels,
                        title, xlabel):
    """Helper: stacked histogram with mean lines + ΣΔ stats box."""
    bins = np.arange(-1.0, 1.0001, 0.05)
    data_list = [deltas_by_bucket[b] for b in bucket_order]
    n_per = [len(d) for d in data_list]
    ax.hist(data_list, bins=bins, stacked=True,
            color=[bucket_colors[b] for b in bucket_order],
            label=[f'{bucket_labels[b]} (n={n})' for b, n in zip(bucket_order, n_per)],
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    all_d = np.concatenate(data_list) if data_list and any(len(d) for d in data_list) else np.array([])
    mu_all = float(all_d.mean()) if len(all_d) else float('nan')
    ax.axvline(mu_all, color='black', linestyle=':', linewidth=1.2, alpha=0.9, label=f'mean all={mu_all:+.3f}')
    for b, c_dark in zip(bucket_order, ['#1e8449', '#117a65', '#a93226', '#a04000', '#5d6d7e']):
        d = deltas_by_bucket[b]
        if len(d):
            mu = float(d.mean())
            ax.axvline(mu, color=c_dark, linestyle=':', linewidth=1.0, alpha=0.85,
                       label=f'mean {b}={mu:+.3f}')
    if len(all_d):
        n_pos = int((all_d > 0).sum()); n_neg = int((all_d < 0).sum()); n_zero = int((all_d == 0).sum())
        sum_pos = float(all_d[all_d > 0].sum()) if n_pos else 0.0
        sum_neg = float(all_d[all_d < 0].sum()) if n_neg else 0.0
        net = sum_pos + sum_neg
        sum_abs = sum_pos - sum_neg
        ax.text(0.02, 0.98,
                f'#Δ>0: {n_pos}  ΣΔ>0: {sum_pos:+.2f}\n'
                f'#Δ<0: {n_neg}  ΣΔ<0: {sum_neg:+.2f}\n'
                f'#Δ=0: {n_zero}\n'
                f'net ΣΔ: {net:+.2f}\n'
                f'total |Δ|: {sum_abs:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.set_xlim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('# questions')
    ax.set_title(title, loc='left', fontsize=9, fontweight='semibold')
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3, linestyle=':')


def plot_l9_base_mode_correct(ax, base_ans, ckpt_ans, golds, ckpt_label='ckpt', title_prefix='L9'):
    """Subset: base mode = gold. Stack by A_sharp (base mm>0.5) / A_diffuse (mm≤0.5)."""
    rows = _l9_l10_compute(base_ans, ckpt_ans, golds)
    a_sharp = np.array([r['d'] for r in rows if r['base_mode_correct'] and r['base_mm'] > 0.5])
    a_diff  = np.array([r['d'] for r in rows if r['base_mode_correct'] and r['base_mm'] <= 0.5])
    _stacked_delta_hist(ax,
        deltas_by_bucket={'A_sharp': a_sharp, 'A_diffuse': a_diff},
        bucket_order=['A_sharp', 'A_diffuse'],
        bucket_colors={'A_sharp': '#16a34a', 'A_diffuse': '#86efac'},
        bucket_labels={'A_sharp': 'A_sharp (base mode=gold, mm>0.5)',
                       'A_diffuse': 'A_diffuse (base mode=gold, mm≤0.5)'},
        title=f"{title_prefix} — Δmass($Y_{{base\\_mode}}$) | base mode CORRECT ({ckpt_label})",
        xlabel=r'$\Delta$mass($Y_{base\_mode}$) = freq$_{ckpt}$ − freq$_{base}$')


def plot_l10_base_mode_wrong(ax, base_ans, ckpt_ans, golds, ckpt_label='ckpt', title_prefix='L10'):
    """Subset: base mode ≠ gold. Stack by B_sharp / B_diffuse / C(base 0/K)."""
    rows = _l9_l10_compute(base_ans, ckpt_ans, golds)
    wrong = [r for r in rows if not r['base_mode_correct']]
    b_sharp = np.array([r['d'] for r in wrong if r['base_c'] > 0 and r['base_mm'] > 0.5])
    b_diff  = np.array([r['d'] for r in wrong if r['base_c'] > 0 and r['base_mm'] <= 0.5])
    c_base  = np.array([r['d'] for r in wrong if r['base_c'] == 0])
    _stacked_delta_hist(ax,
        deltas_by_bucket={'B_sharp': b_sharp, 'B_diffuse': b_diff, 'C_base': c_base},
        bucket_order=['B_sharp', 'B_diffuse', 'C_base'],
        bucket_colors={'B_sharp': '#dc2626', 'B_diffuse': '#fed7aa', 'C_base': '#9ca3af'},
        bucket_labels={'B_sharp': 'B_sharp (mode wrong, base mm>0.5)',
                       'B_diffuse': 'B_diffuse (mode wrong, base mm≤0.5)',
                       'C_base': 'C_base (base 0/K correct)'},
        title=f"{title_prefix} — Δmass($Y_{{base\\_mode}}$) | base mode WRONG ({ckpt_label})",
        xlabel=r'$\Delta$mass($Y_{base\_mode}$) = freq$_{ckpt}$ − freq$_{base}$')


if __name__ == '__main__':
    # Self-test: R16 vs base GSM8K
    import sys
    from pathlib import Path
    ROOT = Path('/mnt/d/fine-tuning')
    base_gsm = ROOT / 'v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json'
    r16_gsm = ROOT / 'v3/E5_grpo/outputs/k128_merged/r16_step42_k128_math.json'

    r_base, k_b, n_b = compute_pass_rates(base_gsm)
    print(f'base GSM8K K=64: K={k_b} n={n_b} mean_rate={r_base.mean():.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_paired_hist(axes[0], r_base, r_base, title='self-test: base vs base', labels=('base', 'base2'))
    plot_transition_10x10(axes[1], r_base, r_base, title='self-test: base vs base')
    plt.tight_layout()
    plt.savefig('/tmp/paper_style_smoke.png', dpi=120)
    print('smoke saved /tmp/paper_style_smoke.png')
