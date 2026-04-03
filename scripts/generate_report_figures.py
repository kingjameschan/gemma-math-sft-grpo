"""
生成实验报告所需的所有图表：
  1. DPO 消融图（带置信区间带）
  2. SFT v2 训练曲线（accuracy vs step）
  3. 错误类型分布对比（SFT vs DPO）
  4. DPO 数据构造统计饼图
  5. 回答长度分布对比

输出到 outputs/figures/ 目录
用法：
  python scripts/generate_report_figures.py
"""
import os, json, glob, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

script_dir  = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.abspath(os.path.join(script_dir, "../outputs"))
figures_dir = os.path.join(outputs_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

def load_eval(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]

def accuracy(records):
    correct = sum(1 for r in records if r['is_correct'])
    return correct / len(records), len(records)

def wilson_ci(n, p, z=1.96):
    """Wilson score confidence interval"""
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - margin, center + margin

def classify_error(d):
    resp = d.get("full_response", "")
    pred = d.get("pred_val", "")
    gold = d.get("gold_val", "")
    if pred is None or pred == "":
        return "No Answer"
    if "####" not in resp:
        return "Format Error"
    try:
        g = float(str(gold).replace(",", ""))
        p = float(str(pred).replace(",", ""))
        if abs(g) > 0 and abs(p - g) / abs(g) < 0.1:
            return "Close Miss"
    except (ValueError, ZeroDivisionError):
        pass
    return "Reasoning Failure"

# ─────────────────────────────────────────────────────────────
# 图1: DPO 消融图（带置信区间带）
# ─────────────────────────────────────────────────────────────
def plot_dpo_ablation():
    print("生成图1: DPO 消融图...")
    # SFT baseline (200样本)
    sft_acc_200, sft_n_200 = 0.755, 200
    # SFT baseline (全量，若已有)
    sft_full_path = os.path.join(outputs_dir, "eval_sft800_full.jsonl")
    sft_full_acc = None
    if os.path.exists(sft_full_path):
        recs = load_eval(sft_full_path)
        if len(recs) >= 1000:
            sft_full_acc, sft_full_n = accuracy(recs)

    pattern = os.path.join(outputs_dir, "eval_dpo_*.jsonl")
    files = sorted(glob.glob(pattern))

    results = []
    for f in files:
        m = re.search(r'eval_dpo_(\w+)\.jsonl', os.path.basename(f))
        label = m.group(1) if m else "?"
        try:
            num_pairs = int(label)
        except ValueError:
            continue
        recs = load_eval(f)
        acc, n = accuracy(recs)
        lo, hi = wilson_ci(n, acc)
        results.append({"label": label, "num_pairs": num_pairs, "acc": acc, "lo": lo, "hi": hi})

    results = sorted(results, key=lambda x: x['num_pairs'])

    fig, ax = plt.subplots(figsize=(9, 5))

    xs = [r['num_pairs'] for r in results]
    ys = [r['acc'] for r in results]
    los = [r['lo'] for r in results]
    his = [r['hi'] for r in results]

    ax.fill_between(xs, los, his, alpha=0.2, color='steelblue', label='95% CI (DPO)')
    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=2, markersize=7, label='DPO accuracy')

    # SFT baseline 线
    sft_lo, sft_hi = wilson_ci(sft_n_200, sft_acc_200)
    ax.axhline(sft_acc_200, color='orange', linestyle='--', linewidth=1.5,
               label=f'SFT-800 baseline ({sft_acc_200:.1%}, n=200)')
    ax.fill_between([min(xs)-10, max(xs)+10], sft_lo, sft_hi, alpha=0.15, color='orange')

    if sft_full_acc:
        ax.axhline(sft_full_acc, color='red', linestyle=':', linewidth=1.5,
                   label=f'SFT-800 full ({sft_full_acc:.1%}, n={sft_full_n})')

    for r in results:
        ax.annotate(f"{r['acc']:.1%}", (r['num_pairs'], r['acc']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Number of DPO Pairs', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DPO Data Quantity vs Accuracy (GSM8K, 200 test samples)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.60, 0.90)
    ax.set_xlim(left=-10)

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig1_dpo_ablation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─────────────────────────────────────────────────────────────
# 图2: SFT v2 训练曲线
# ─────────────────────────────────────────────────────────────
def plot_sft_curve():
    print("生成图2: SFT 训练曲线...")
    pattern = os.path.join(outputs_dir, "eval_details_v2_*.jsonl")
    files = glob.glob(pattern)

    points = []
    for f in files:
        m = re.search(r'eval_details_v2_(\d+)\.jsonl', os.path.basename(f))
        if not m:
            continue
        step = int(m.group(1))
        recs = load_eval(f)
        acc, n = accuracy(recs)
        lo, hi = wilson_ci(n, acc)
        points.append({"step": step, "acc": acc, "lo": lo, "hi": hi, "n": n})

    points.sort(key=lambda x: x['step'])
    if not points:
        print("  未找到 eval_details_v2_*.jsonl 文件，跳过")
        return

    xs = [p['step'] for p in points]
    ys = [p['acc'] for p in points]
    los = [p['lo'] for p in points]
    his = [p['hi'] for p in points]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(xs, los, his, alpha=0.2, color='steelblue')
    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=2, markersize=5, label='SFT v2 accuracy')

    # 标注最佳点
    best = max(points, key=lambda p: p['acc'])
    ax.annotate(f"Best: {best['acc']:.1%}\n(step {best['step']})",
                (best['step'], best['acc']),
                textcoords="offset points", xytext=(15, -25),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('SFT v2 Training Curve (GSM8K, 50 test samples per checkpoint)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig2_sft_training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─────────────────────────────────────────────────────────────
# 图3: 错误类型分布对比
# ─────────────────────────────────────────────────────────────
def plot_error_types():
    print("生成图3: 错误类型分布...")
    configs = [
        ("SFT-800", os.path.join(outputs_dir, "eval_details_sft800_200.jsonl")),
        ("DPO-100", os.path.join(outputs_dir, "eval_dpo_100.jsonl")),
        ("DPO-400", os.path.join(outputs_dir, "eval_dpo_400.jsonl")),
    ]

    all_cats = ["Format Error", "No Answer", "Close Miss", "Reasoning Failure"]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db']
    x = np.arange(len(all_cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, path) in enumerate(configs):
        if not os.path.exists(path):
            continue
        recs = load_eval(path)
        errors = [r for r in recs if not r['is_correct'] and r.get('full_response')]
        if not errors:
            continue
        cats = defaultdict(int)
        for r in errors:
            cats[classify_error(r)] += 1
        total = len(errors)
        vals = [cats[c] / total * 100 for c in all_cats]
        bars = ax.bar(x + i*width, vals, width, label=f'{name} (n={total})')

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('% of Errors', fontsize=12)
    ax.set_title('Error Type Distribution: SFT vs DPO', fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_cats)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig3_error_types.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─────────────────────────────────────────────────────────────
# 图4: DPO 数据构造统计饼图
# ─────────────────────────────────────────────────────────────
def plot_dpo_data_stats():
    print("生成图4: DPO 数据构造统计...")
    stats_path = os.path.join(os.path.dirname(outputs_dir), "data/dpo/generation_stats_full.json")
    if not os.path.exists(stats_path):
        print("  未找到 generation_stats_full.json，跳过")
        return

    stats = json.load(open(stats_path, encoding='utf-8'))
    total_q = stats['total_questions']
    pairs = stats['total_pairs']
    all_correct = stats['skipped_all_correct']
    no_correct = stats['skipped_no_correct']
    remaining = total_q - pairs - all_correct - no_correct

    labels = [
        f'Valid Pairs\n({pairs}, {pairs/total_q*100:.1f}%)',
        f'All Correct\n({all_correct}, {all_correct/total_q*100:.1f}%)',
        f'All Wrong\n({no_correct}, {no_correct/total_q*100:.1f}%)',
        f'Other Skip\n({remaining}, {remaining/total_q*100:.1f}%)',
    ]
    sizes = [pairs, all_correct, no_correct, remaining]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    explode = (0.05, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='', startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'DPO Data Construction\n(from {total_q} training questions)', fontsize=13)

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig4_dpo_data_stats.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


# ─────────────────────────────────────────────────────────────
# 图5: 回答长度分布
# ─────────────────────────────────────────────────────────────
def plot_response_length():
    print("生成图5: 回答长度分布...")
    configs = [
        ("DPO-100", os.path.join(outputs_dir, "eval_dpo_100.jsonl"), '#3498db'),
        ("DPO-400", os.path.join(outputs_dir, "eval_dpo_400.jsonl"), '#e74c3c'),
        ("DPO-800", os.path.join(outputs_dir, "eval_dpo_800.jsonl"), '#2ecc71'),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    has_data = False
    for name, path, color in configs:
        if not os.path.exists(path):
            continue
        recs = load_eval(path)
        lengths = [len(r['full_response'].split()) for r in recs if r.get('full_response')]
        if lengths:
            ax.hist(lengths, bins=40, alpha=0.5, color=color,
                    label=f'{name} (μ={np.mean(lengths):.0f})', density=True)
            has_data = True

    if not has_data:
        print("  无 full_response 数据，跳过")
        return

    ax.set_xlabel('Response Length (words)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Response Length Distribution: DPO Variants', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig5_response_length.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  保存: {out}")


if __name__ == "__main__":
    plot_dpo_ablation()
    plot_sft_curve()
    plot_error_types()
    plot_dpo_data_stats()
    plot_response_length()
    print(f"\n全部图表已保存至: {figures_dir}")
