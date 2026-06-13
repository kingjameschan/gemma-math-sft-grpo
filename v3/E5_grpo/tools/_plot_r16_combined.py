"""R16 Clean GRPO combined figure — step_42 dev-peak ckpt.

Panels (2x3):
  L1.1a — GSM8K test pass@K + maj@K (R16 step_42 vs base IT) [K=64]
  L1.1b — MATH numeric pass@K + maj@K (R16 step_42 vs base IT) [if available]
  L2.1  — Dev acc curve across 32 ckpts (R16 step_1..60, save every 2 steps)
  L2.2  — Train reward/mean over 60 steps
  L2.3  — Train response_length/mean over 60 steps
  L3.1  — Summary text (step42 test/dev acc, training hyperparams)

Inputs:
  v3/E5_grpo/outputs/k64_r16_step42/r16_step42_k64_gsm8k.json
  v3/E5_grpo/outputs/k64_r16_step42/r16_step42_k64_math.json (optional)
  v3/E5_grpo/outputs/r16_dev_eval/r16_dev_eval_step1-60.json
  v3/E5_grpo/outputs/baseit_r16_clean_grpo_logs/train.log
  v3/E5_grpo/outputs/eval_r16_step42/eval_step42_test_results.json
  v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json
  v3/E1_baseline/outputs/pass_at_k_math_20260513_092839/base_gemma-2-2b-it_k64.json
"""
import json, math, re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
R16_K64_GSM = ROOT / "v3/E5_grpo/outputs/k64_r16_step42/r16_step42_k64_gsm8k.json"
R16_K64_MATH = ROOT / "v3/E5_grpo/outputs/k64_r16_step42/r16_step42_k64_math.json"
R16_DEV_EVAL = ROOT / "v3/E5_grpo/outputs/r16_dev_eval/r16_dev_eval_step1-60.json"
R16_TRAIN_LOG = ROOT / "v3/E5_grpo/outputs/baseit_r16_clean_grpo_logs/train.log"
R16_TEST_PASS1 = ROOT / "v3/E5_grpo/outputs/eval_r16_step42/eval_step42_test_results.json"
BASE_K64_GSM = ROOT / "v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64.json"
BASE_K64_MATH = ROOT / "v3/E1_baseline/outputs/pass_at_k_math_20260513_092839/base_gemma-2-2b-it_k64.json"

OUT_DIR = ROOT / "v3/E5_grpo/outputs/k64_r16_step42"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "r16_step42_combined.png"

KS = [1, 2, 4, 8, 16, 32, 64]
TAG = "R16 step_42"
COLOR_BASE = "black"
COLOR_R16 = "#7c3aed"  # purple for GRPO (distinct from DAPO red, SFT green)


def pass_at_k_unbiased(c, n, k):
    """Unbiased pass@k from c correct out of n samples."""
    if c == 0: return 0.0
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def majority_vote(preds):
    """Return most-common non-empty pred, breaking ties by first occurrence."""
    counts = {}
    for p in preds:
        if not p: continue
        counts[p] = counts.get(p, 0) + 1
    if not counts: return ""
    return max(counts.items(), key=lambda x: (x[1], -preds.index(x[0])))[0]


def passk_majk_from_r16(json_path):
    """R16 schema: per_q[i] = {gold, samples:[{c,a,L}]}"""
    if not Path(json_path).exists(): return None, None
    d = json.load(open(json_path))
    per_q = d["per_q"]
    n = len(per_q[0]["samples"])
    passk = {}
    majk = {}
    for k in KS:
        # pass@k: per-question unbiased
        pks = [pass_at_k_unbiased(sum(s["c"] for s in q["samples"]), n, k)
               for q in per_q]
        passk[k] = 100.0 * sum(pks) / len(pks)
        # maj@k: use first k samples to majority vote
        correct_majk = 0
        for q in per_q:
            preds = [str(s["a"]) for s in q["samples"][:k]]
            mp = majority_vote(preds)
            if mp and mp == str(q["gold"]).strip():
                correct_majk += 1
        majk[k] = 100.0 * correct_majk / len(per_q)
    return passk, majk


def passk_majk_from_base(json_path):
    """Base K=64 schema: samples[i].any_correct_per_K is INT count, any_preds is K-list."""
    if not Path(json_path).exists(): return None, None
    d = json.load(open(json_path))
    samples = d["samples"]
    n = d["config"]["K"]
    passk = {}
    majk = {}
    for k in KS:
        pks = [pass_at_k_unbiased(s["any_correct_per_K"], n, k) for s in samples]
        passk[k] = 100.0 * sum(pks) / len(pks)
        correct = 0
        for s in samples:
            preds = [str(p) for p in s["any_preds"][:k]]
            mp = majority_vote(preds)
            if mp and mp == str(s["gold"]).strip():
                correct += 1
        majk[k] = 100.0 * correct / len(samples)
    return passk, majk


def parse_train_log(log_path):
    """Extract per-step reward/mean and response_length/mean from train.log."""
    if not Path(log_path).exists(): return [], [], []
    steps, rewards, lens = [], [], []
    re_step = re.compile(r"step:(\d+)\b")
    re_reward = re.compile(r"critic/rewards/mean:([\d.eE+-]+)")
    re_len = re.compile(r" response_length/mean:([\d.eE+-]+)")
    seen = set()
    for line in open(log_path):
        ms = re_step.search(line)
        mr = re_reward.search(line)
        ml = re_len.search(line)
        if ms and mr and ml:
            s = int(ms.group(1))
            if s in seen: continue
            seen.add(s)
            steps.append(s); rewards.append(float(mr.group(1))); lens.append(float(ml.group(1)))
    order = np.argsort(steps)
    return [steps[i] for i in order], [rewards[i] for i in order], [lens[i] for i in order]


# ============ Load ============
print(f"[load] R16 K=64 GSM8K: {R16_K64_GSM.exists()}")
r16_gsm_passk, r16_gsm_majk = passk_majk_from_r16(R16_K64_GSM)
print(f"  step_42 GSM8K pass@1={r16_gsm_passk[1]:.2f} pass@64={r16_gsm_passk[64]:.2f}")

print(f"[load] R16 K=64 MATH: {R16_K64_MATH.exists()}")
r16_math_passk, r16_math_majk = passk_majk_from_r16(R16_K64_MATH)

print(f"[load] base K=64 GSM8K: {BASE_K64_GSM.exists()}")
base_gsm_passk, base_gsm_majk = passk_majk_from_base(BASE_K64_GSM)

print(f"[load] base K=64 MATH: {BASE_K64_MATH.exists()}")
base_math_passk, base_math_majk = passk_majk_from_base(BASE_K64_MATH)

dev_eval = json.load(open(R16_DEV_EVAL)) if R16_DEV_EVAL.exists() else []
print(f"[load] R16 dev eval: {len(dev_eval)} ckpts")

train_steps, train_rewards, train_lens = parse_train_log(R16_TRAIN_LOG)
print(f"[load] R16 train log: {len(train_steps)} steps")

test_pass1 = json.load(open(R16_TEST_PASS1)) if R16_TEST_PASS1.exists() else None

# ============ Render ============
fig = plt.figure(figsize=(18, 10), facecolor="white")
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.42)

# ----- L1.1a — GSM8K pass@K -----
ax = fig.add_subplot(gs[0, 0])
ax.plot(KS, [base_gsm_passk[k] for k in KS], "-", color=COLOR_BASE, lw=2, label="base pass@K")
ax.plot(KS, [base_gsm_majk[k] for k in KS], "--", color=COLOR_BASE, lw=1.5, alpha=0.7, label="base maj@K")
ax.plot(KS, [r16_gsm_passk[k] for k in KS], "-", color=COLOR_R16, lw=2, label=f"{TAG} pass@K")
ax.plot(KS, [r16_gsm_majk[k] for k in KS], "--", color=COLOR_R16, lw=1.5, alpha=0.85, label=f"{TAG} maj@K")
ax.set_xscale("log", base=2)
ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
ax.set_title(
    f"L1.1a — GSM8K test (n=1319) K=64 T=0.7\n"
    f"base p@1={base_gsm_passk[1]:.1f} p@64={base_gsm_passk[64]:.1f} maj@64={base_gsm_majk[64]:.1f}\n"
    f"{TAG} p@1={r16_gsm_passk[1]:.1f} p@64={r16_gsm_passk[64]:.1f} maj@64={r16_gsm_majk[64]:.1f}",
    loc="left", fontsize=9, fontweight="semibold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
_y = ([base_gsm_passk[k] for k in KS] + [base_gsm_majk[k] for k in KS] +
      [r16_gsm_passk[k] for k in KS] + [r16_gsm_majk[k] for k in KS])
ax.set_ylim(math.floor(min(_y) / 5) * 5, math.ceil(max(_y) / 5) * 5)

# ----- L1.1b — MATH pass@K -----
ax = fig.add_subplot(gs[0, 1])
title_m = (f"base p@1={base_math_passk[1]:.1f} p@64={base_math_passk[64]:.1f} "
           f"maj@64={base_math_majk[64]:.1f}" if base_math_passk else "base 待加载")
if base_math_passk is not None:
    ax.plot(KS, [base_math_passk[k] for k in KS], "-", color=COLOR_BASE, lw=2, label="base pass@K")
    ax.plot(KS, [base_math_majk[k] for k in KS], "--", color=COLOR_BASE, lw=1.5, alpha=0.7, label="base maj@K")
if r16_math_passk is not None:
    ax.plot(KS, [r16_math_passk[k] for k in KS], "-", color=COLOR_R16, lw=2, label=f"{TAG} pass@K")
    ax.plot(KS, [r16_math_majk[k] for k in KS], "--", color=COLOR_R16, lw=1.5, alpha=0.85, label=f"{TAG} maj@K")
    title_m += (f"\n{TAG} p@1={r16_math_passk[1]:.1f} p@64={r16_math_passk[64]:.1f} "
                f"maj@64={r16_math_majk[64]:.1f}")
else:
    title_m += f"\n{TAG} K=64 跑中 (greedy p@1={test_pass1['math_numeric']['acc']:.2f}" + (
        " % via /eval_r16_step42)" if test_pass1 else ")")
    ax.text(0.5, 0.4, "MATH K=64\n跑中 ETA ~30min", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color=COLOR_R16, alpha=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel("K"); ax.set_ylabel("accuracy (%)")
ax.set_title(f"L1.1b — MATH numeric (n=2927) K=64 T=0.7\n{title_m}",
             loc="left", fontsize=9, fontweight="semibold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# ----- L2.1 — Dev acc over 32 ckpts -----
ax = fig.add_subplot(gs[0, 2])
if dev_eval:
    steps = [d["step"] for d in dev_eval]
    accs = [d["acc"] for d in dev_eval]
    best_i = int(np.argmax(accs))
    ax.plot(steps, accs, "-o", color=COLOR_R16, lw=2, ms=4, label="dev acc (n=500)")
    ax.scatter([steps[best_i]], [accs[best_i]], s=120, color="gold",
               edgecolors="black", zorder=5, label=f"best step_{steps[best_i]}={accs[best_i]:.2f}%")
    ax.axhline(61.33, color="grey", ls=":", alpha=0.7, label="base 61.33%")
ax.set_xlabel("training step"); ax.set_ylabel("dev GSM8K acc (%)")
ax.set_title(f"L2.1 — Dev acc sweep (32 ckpts, save every 2 steps)\n"
             f"step_{steps[best_i]} 选作 best (acc={accs[best_i]:.2f}%)",
             loc="left", fontsize=9, fontweight="semibold")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

# ----- L2.2 — Train reward over steps -----
ax = fig.add_subplot(gs[1, 0])
if train_steps:
    ax.plot(train_steps, train_rewards, "-", color=COLOR_R16, lw=2)
    ax.axhline(train_rewards[0], color="grey", ls=":", alpha=0.6,
               label=f"start {train_rewards[0]:.3f}")
    peak_i = int(np.argmax(train_rewards))
    ax.scatter([train_steps[peak_i]], [train_rewards[peak_i]], s=80, color="gold",
               edgecolors="black", zorder=5,
               label=f"peak step_{train_steps[peak_i]}={train_rewards[peak_i]:.3f}")
ax.set_xlabel("step"); ax.set_ylabel("critic/rewards/mean")
ax.set_title("L2.2 — Train reward (G=8, T=1.0, KL=0)",
             loc="left", fontsize=9, fontweight="semibold")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

# ----- L2.3 — Train response_length -----
ax = fig.add_subplot(gs[1, 1])
if train_steps:
    ax.plot(train_steps, train_lens, "-", color=COLOR_R16, lw=2)
    ax.axhline(train_lens[0], color="grey", ls=":", alpha=0.6,
               label=f"start {train_lens[0]:.0f}")
ax.set_xlabel("step"); ax.set_ylabel("response_length/mean (token)")
ax.set_title(f"L2.3 — Train response_length (max=512)\n"
             f"start={train_lens[0]:.0f} → end={train_lens[-1]:.0f} (Δ={train_lens[-1] - train_lens[0]:+.0f})",
             loc="left", fontsize=9, fontweight="semibold")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)

# ----- L3.1 — Summary text -----
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary_lines = [
    f"R16 Clean GRPO — Gemma2-2B-IT base",
    f"4 项 DAPO additions OFF (vs R15):",
    f"  · Clip-Higher  ε_high=0.40 → 0.20",
    f"  · filter_groups=True → False",
    f"  · overlong_buffer=True → False",
    f"  · loss_agg token-mean → seq-mean-token-mean",
    f"",
    f"Hyper: lr=2e-5  G=8  bs=384  μ=3  KL=0",
    f"       max_resp=512  warmup=10  epoch=2",
    f"",
    f"Training: 60 steps × ~17 min = 16.9 h on L20 (Aliyun)",
    f"          reward {train_rewards[0]:.3f} → {train_rewards[-1]:.3f}",
    f"          response_length {train_lens[0]:.0f} → {train_lens[-1]:.0f}",
    f"",
    f"Dev sweep best: step_{steps[best_i]} = {accs[best_i]:.2f}% (32 ckpts saved)",
    f"",
    f"step_42 test pass@1:",
    f"  GSM8K (1319): {test_pass1['gsm8k_test']['acc']:.2f}% (greedy T=0) | K=64 p@1={r16_gsm_passk[1]:.2f}",
    f"  MATH  (2927): {test_pass1['math_numeric']['acc']:.2f}% (greedy T=0) | K=64 p@1={'待加载' if not r16_math_passk else f'{r16_math_passk[1]:.2f}'}",
    f"",
    f"K=64 vs base GSM8K:",
    f"  pass@1: {base_gsm_passk[1]:.1f} → {r16_gsm_passk[1]:.1f}  (Δ={r16_gsm_passk[1] - base_gsm_passk[1]:+.1f})",
    f"  pass@64: {base_gsm_passk[64]:.1f} → {r16_gsm_passk[64]:.1f}  (Δ={r16_gsm_passk[64] - base_gsm_passk[64]:+.1f})",
    f"  maj@64: {base_gsm_majk[64]:.1f} → {r16_gsm_majk[64]:.1f}  (Δ={r16_gsm_majk[64] - base_gsm_majk[64]:+.1f})",
]
ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
        ha="left", va="top", family="monospace", fontsize=9)

fig.suptitle(f"R16 Clean GRPO — step_{steps[best_i]} dev-peak combined view",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nsaved: {OUT_FILE}")
