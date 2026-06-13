"""R16 step_42 paper-style 2-panel: paired histogram + 10x10 transition.

Layout: 2 rows × 2 cols.
  Row 1: GSM8K — paired hist (left) + 10x10 transition (right)
  Row 2: MATH numeric — paired hist (left) + 10x10 transition (right)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _paper_style_panels import compute_pass_rates, plot_paired_hist, plot_transition_10x10

ROOT = Path('/mnt/d/fine-tuning')
BASE_GSM_K64 = ROOT / 'v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json'
BASE_MATH_K64 = ROOT / 'v3/E1_baseline/outputs/pass_at_k_math_20260513_092839/base_gemma-2-2b-it_k64.json'
R16_GSM_K64 = ROOT / 'v3/E5_grpo/outputs/k64_r16_step42/r16_step42_k64_gsm8k.json'
R16_MATH_K128 = ROOT / 'v3/E5_grpo/outputs/k128_merged/r16_step42_k128_math.json'
OUT = ROOT / 'v3/E5_grpo/tools/figures/r16_step42_paper_panels.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Load all
r_base_gsm, K_b_g, n_b_g = compute_pass_rates(BASE_GSM_K64)
r_r16_gsm, K_r_g, n_r_g = compute_pass_rates(R16_GSM_K64)
r_base_math, K_b_m, n_b_m = compute_pass_rates(BASE_MATH_K64)
r_r16_math, K_r_m, n_r_m = compute_pass_rates(R16_MATH_K128)

print(f'GSM8K: base K={K_b_g} mean={r_base_gsm.mean():.3f}, R16 K={K_r_g} mean={r_r16_gsm.mean():.3f}')
print(f'MATH:  base K={K_b_m} mean={r_base_math.mean():.3f}, R16 K={K_r_m} mean={r_r16_math.mean():.3f}')

# 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_paired_hist(axes[0, 0], r_base_gsm, r_r16_gsm,
                 title=f'GSM8K (n={n_r_g}) — per-q pass-rate distribution\nbase K={K_b_g} vs R16-step42 K={K_r_g}',
                 labels=(f'base (K={K_b_g})', f'R16-step42 (K={K_r_g})'))
plot_transition_10x10(axes[0, 1], r_base_gsm, r_r16_gsm,
                      title='GSM8K — base→R16 pass-rate transition (log color)')

plot_paired_hist(axes[1, 0], r_base_math, r_r16_math,
                 title=f'MATH numeric (n={n_r_m}) — per-q pass-rate distribution\nbase K={K_b_m} vs R16-step42 K={K_r_m}',
                 labels=(f'base (K={K_b_m})', f'R16-step42 (K={K_r_m})'))
plot_transition_10x10(axes[1, 1], r_base_math, r_r16_math,
                      title='MATH — base→R16 pass-rate transition (log color)')

plt.suptitle('R16 step_42 GRPO — capability shift (paper §4.1 style)\nbase Gemma2-2B-IT vs R16 step_42 GRPO ckpt',
             fontsize=12, y=1.005)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f'\n=== saved → {OUT}')
