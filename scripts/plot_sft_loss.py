import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logs = {
    "lr=3e-4 (batch=1)": "D:/fine-tuning/logs/sft_1.7b_lr3e4_r16.log",
    "lr=2e-4 (batch=1)": "D:/fine-tuning/logs/sft_1.7b_lr2e4_r16.log",
    "lr=1e-4 (batch=6)": "D:/fine-tuning/logs/sft_1.7b_lr1e4_r16.log",
}

train_data = {}
eval_data  = {}

for name, path in logs.items():
    train_data[name] = []
    eval_data[name]  = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # train loss: 'loss': X, ... 'epoch': Y
            m = re.search(r"'loss':\s*([0-9.]+).*?'epoch':\s*([0-9.]+)", line)
            if m and "eval_loss" not in line and "train_runtime" not in line:
                train_data[name].append((float(m.group(2)), float(m.group(1))))
            # eval loss
            m2 = re.search(r"'eval_loss':\s*([0-9.]+).*?'epoch':\s*([0-9.]+)", line)
            if m2:
                eval_data[name].append((float(m2.group(2)), float(m2.group(1))))

for name in logs:
    print(f"{name}: {len(train_data[name])} train pts, {len(eval_data[name])} eval pts")

colors = {"lr=3e-4 (batch=1)": "#e74c3c",
          "lr=2e-4 (batch=1)": "#3498db",
          "lr=1e-4 (batch=6)": "#2ecc71"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Qwen3-1.7B SFT — Loss Curves (Phase A: lr search, r=16)", fontsize=13)

def ema(values, alpha=0.3):
    smoothed, v = [], None
    for x in values:
        v = x if v is None else alpha * x + (1 - alpha) * v
        smoothed.append(v)
    return smoothed

ax = axes[0]
for name, pts in train_data.items():
    if not pts:
        continue
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, color=colors[name], linewidth=0.8, alpha=0.3)
    ax.plot(xs, ema(ys), label=name, color=colors[name], linewidth=2.0)
ax.set_title("Train Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax = axes[1]
for name, pts in eval_data.items():
    if not pts:
        continue
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, marker="o", markersize=5, label=name,
            color=colors[name], linewidth=1.5)
    best_ep, best_loss = min(zip(xs, ys), key=lambda x: x[1])
    ax.annotate(f"{best_loss:.3f}", (best_ep, best_loss),
                textcoords="offset points", xytext=(5, -12),
                fontsize=8, color=colors[name])
ax.set_title("Eval Loss (200-sample set)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

plt.tight_layout()
out = "D:/fine-tuning/outputs/sft_1.7b_phase_a_loss.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
