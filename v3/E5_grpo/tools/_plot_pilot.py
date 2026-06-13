"""Pilot run analysis: lr=5e-5 b=0.04 G=8 T=0.7 max_steps=150.

Inputs:
  - v3/E5_grpo/outputs/pilot/eval/*.json    (5 ckpts × pass@1)
  - v3/E5_grpo/outputs/pilot/logs/train.log (per-step reward/KL/loss/grad_norm)

Output:
  - v3/E5_grpo/outputs/pilot/pilot_summary.png — 6 panels
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "v3" / "E5_grpo" / "outputs" / "pilot"
OUT = PILOT / "pilot_summary.png"


def parse_train(log_path):
    metrics = {
        "loss": [], "grad_norm": [], "kl": [], "entropy": [],
        "reward": [], "reward_std": [], "frac_reward_zero_std": [],
        "completions/mean_length": [], "step_time": [],
    }
    text = log_path.read_text()
    for line in text.splitlines():
        if not (line.startswith("{'loss'") or line.startswith("{ 'loss'")):
            continue
        for k in metrics:
            # Handle both quoted-string and bare-numeric formats
            for pattern in [
                rf"'{re.escape(k)}': '([^']+)'",
                rf"'{re.escape(k)}': ([0-9eE.+-]+)",
            ]:
                m = re.search(pattern, line)
                if m:
                    try:
                        metrics[k].append(float(m.group(1)))
                    except ValueError:
                        pass
                    break
    return metrics


def load_eval():
    """Returns list of (step, pass1, boxed, mean_len)."""
    out = []
    for f in sorted(PILOT.glob("eval/*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        d = json.loads(f.read_text())
        step = d["config"]["step"]
        out.append({
            "step": step,
            "pass1": d["pass_at_1"],
            "boxed": d["boxed_rate"],
            "mean_len": d["mean_response_length"],
        })
    return out


def main():
    print("[load] train log...")
    train = parse_train(PILOT / "logs" / "train.log")
    n = len(train["loss"])
    steps = list(range(1, n + 1))
    print(f"  parsed {n} train steps")

    print("[load] eval JSONs...")
    evals = load_eval()
    eval_steps = [e["step"] for e in evals]
    eval_pass1 = [e["pass1"] * 100 for e in evals]
    eval_boxed = [e["boxed"] * 100 for e in evals]
    eval_mlen = [e["mean_len"] for e in evals]
    print(f"  eval steps: {eval_steps}")
    print(f"  pass@1:     {[f'{p:.2f}' for p in eval_pass1]}")
    print(f"  boxed:      {[f'{b:.1f}' for b in eval_boxed]}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Reward over training steps
    ax = axes[0, 0]
    if train["reward"]:
        ax.plot(steps, train["reward"], "o", color="C0", alpha=0.4, ms=3,
                label=f"per-step (n={n})")
        # Rolling mean window=10
        w = 10
        if n >= w:
            roll = np.convolve(train["reward"], np.ones(w)/w, mode="valid")
            ax.plot(range(w, n + 1), roll, "-", color="C0", lw=2.5,
                    label=f"rolling mean (w={w})")
    ax.axhline(np.mean(train["reward"][:20]), color="gray", ls=":",
               label=f"first-20 mean = {np.mean(train['reward'][:20]):.3f}")
    ax.axhline(np.mean(train["reward"][-20:]), color="red", ls=":",
               label=f"last-20 mean = {np.mean(train['reward'][-20:]):.3f}")
    ax.set_title("Train reward (per-step + rolling mean)")
    ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 2. KL trajectory
    ax = axes[0, 1]
    if train["kl"]:
        ax.plot(steps, train["kl"], "-", color="C2", lw=1.5)
        ax.axhline(0.05, color="orange", ls="--", alpha=0.5, label="KL=0.05 (mod)")
        ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="KL=0.5 (high)")
    ax.set_title("KL divergence vs ref")
    ax.set_xlabel("step"); ax.set_ylabel("KL")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. frac_reward_zero_std (degenerate batches)
    ax = axes[0, 2]
    if train["frac_reward_zero_std"]:
        ax.plot(steps, train["frac_reward_zero_std"], "-", color="C3", lw=1.5)
        # Rolling mean
        w = 10
        if n >= w:
            roll = np.convolve(train["frac_reward_zero_std"],
                               np.ones(w)/w, mode="valid")
            ax.plot(range(w, n + 1), roll, "-", color="darkred", lw=2,
                    label=f"rolling mean (w={w})")
    ax.set_title("frac_reward_zero_std (degenerate batches)")
    ax.set_xlabel("step"); ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. Loss + grad_norm (twin axis)
    ax = axes[1, 0]
    if train["loss"]:
        l1 = ax.plot(steps, train["loss"], "-", color="C4", lw=1.5, label="loss")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylabel("loss (symlog)", color="C4")
        ax.tick_params(axis="y", labelcolor="C4")
    ax2 = ax.twinx()
    if train["grad_norm"]:
        l2 = ax2.plot(steps, train["grad_norm"], "-", color="C5", lw=1.5,
                      alpha=0.7, label="grad_norm")
        ax2.set_yscale("symlog", linthresh=1e-3)
        ax2.set_ylabel("grad_norm (symlog)", color="C5")
        ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("Loss + grad_norm")
    ax.set_xlabel("step"); ax.grid(alpha=0.3)

    # 5. Completion length
    ax = axes[1, 1]
    if train["completions/mean_length"]:
        ax.plot(steps, train["completions/mean_length"], "-", color="C6", lw=1.5)
    if eval_mlen:
        ax.plot(eval_steps, eval_mlen, "rs", ms=10, label="D_dev mean_len at ckpt")
    ax.set_title("Completion length (train rollouts + eval)")
    ax.set_xlabel("step"); ax.set_ylabel("tokens")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6. Eval pass@1 + boxed_rate over ckpts
    ax = axes[1, 2]
    if eval_pass1:
        ax.plot(eval_steps, eval_pass1, "o-", color="C0", ms=10, lw=2,
                label="D_dev pass@1")
        ax.plot(eval_steps, eval_boxed, "s--", color="C1", ms=10, lw=2,
                label="D_dev boxed_rate")
        # E1 base IT references
        ax.axhline(62, color="C0", ls=":", alpha=0.5, label="base IT pass@1≈62%")
        ax.axhline(46, color="C1", ls=":", alpha=0.5, label="base IT boxed≈46%")
        for x, y in zip(eval_steps, eval_pass1):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color="C0")
        for x, y in zip(eval_steps, eval_boxed):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=9, color="C1")
    ax.set_title("D_dev eval (pass@1 + boxed_rate per ckpt)")
    ax.set_xlabel("step"); ax.set_ylabel("%")
    ax.set_ylim(20, 100)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("GRPO Pilot — lr=5e-5  β=0.04  G=8  T=0.7  max_steps=150  (39 min on L40S)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
