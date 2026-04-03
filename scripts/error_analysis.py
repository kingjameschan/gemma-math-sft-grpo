"""
错误分析脚本：对比 SFT-800 vs DPO 变体的错误模式
分析：
  1. Overlap 分析（SFT错/DPO对，SFT对/DPO错）
  2. 错误分类（格式错误/计算偏差/推理失败/无答案）
  3. 答案大小分组准确率（难度分析）
  4. 回答长度分布
  5. McNemar's test 统计显著性
用法：
  python scripts/error_analysis.py
  python scripts/error_analysis.py --sft outputs/eval_sft800_full.jsonl --dpo outputs/eval_dpo_100.jsonl
"""
import os, json, re, argparse
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.abspath(os.path.join(script_dir, "../outputs"))

parser = argparse.ArgumentParser()
parser.add_argument("--sft", default=os.path.join(output_dir, "eval_details_sft800_200.jsonl"))
parser.add_argument("--dpo", default=os.path.join(output_dir, "eval_dpo_100.jsonl"))
parser.add_argument("--dpo_name", default="dpo_100")
args = parser.parse_args()


def load(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = d["question"]
            data[key] = d
    return data


def classify_error(d):
    """对错误样本分类"""
    resp = d.get("full_response", "")
    gold = d.get("gold_val", "")
    pred = d.get("pred_val", "")

    if pred is None or pred == "":
        return "no_answer"
    if "####" not in resp:
        return "format_error"
    try:
        g = float(str(gold).replace(",", ""))
        p = float(str(pred).replace(",", ""))
        if abs(g) > 0 and abs(p - g) / abs(g) < 0.1:
            return "close_miss"
    except (ValueError, ZeroDivisionError):
        pass
    return "reasoning_failure"


def answer_bucket(gold_val):
    try:
        v = abs(float(str(gold_val).replace(",", "")))
        if v <= 10:    return "0-10"
        if v <= 100:   return "11-100"
        if v <= 1000:  return "101-1000"
        return "1000+"
    except ValueError:
        return "unknown"


def response_length(d):
    return len(d.get("full_response", "").split())


print(f"\n{'='*60}")
print(f"SFT 文件: {os.path.basename(args.sft)}")
print(f"DPO 文件: {os.path.basename(args.dpo)} ({args.dpo_name})")
print('='*60)

sft_data = load(args.sft)
dpo_data = load(args.dpo)

# 共同题目
common_keys = set(sft_data) & set(dpo_data)
print(f"\n共同评测题目: {len(common_keys)} 题")

# ── 1. Overlap 分析 ──────────────────────────────────────────
sft_wrong_dpo_right = []
sft_right_dpo_wrong = []
both_right = []
both_wrong = []

for q in common_keys:
    s = sft_data[q]["is_correct"]
    d = dpo_data[q]["is_correct"]
    if not s and d:     sft_wrong_dpo_right.append(q)
    elif s and not d:   sft_right_dpo_wrong.append(q)
    elif s and d:       both_right.append(q)
    else:               both_wrong.append(q)

n = len(common_keys)
print(f"\n── 1. Overlap 分析 ({'共'+str(n)+'题'}) ──")
print(f"  两者都对:          {len(both_right):4d} ({len(both_right)/n*100:.1f}%)")
print(f"  SFT错 DPO对(修复): {len(sft_wrong_dpo_right):4d} ({len(sft_wrong_dpo_right)/n*100:.1f}%)")
print(f"  SFT对 DPO错(退步): {len(sft_right_dpo_wrong):4d} ({len(sft_right_dpo_wrong)/n*100:.1f}%)")
print(f"  两者都错:          {len(both_wrong):4d} ({len(both_wrong)/n*100:.1f}%)")
net = len(sft_wrong_dpo_right) - len(sft_right_dpo_wrong)
print(f"  净修复: {net:+d} 题")

# 修复案例样本
if sft_wrong_dpo_right:
    print(f"\n  [修复案例 Top3]")
    for q in list(sft_wrong_dpo_right)[:3]:
        sd = sft_data[q]; dd = dpo_data[q]
        print(f"  Q: {q[:80]}...")
        print(f"     SFT预测={sd['pred_val']} DPO预测={dd['pred_val']} 正确={sd['gold_val']}")

if sft_right_dpo_wrong:
    print(f"\n  [退步案例 Top3]")
    for q in list(sft_right_dpo_wrong)[:3]:
        sd = sft_data[q]; dd = dpo_data[q]
        print(f"  Q: {q[:80]}...")
        print(f"     SFT预测={sd['pred_val']} DPO预测={dd['pred_val']} 正确={sd['gold_val']}")

# ── 2. 错误分类 ──────────────────────────────────────────────
print(f"\n── 2. 错误分类 ──")
for name, data in [("SFT-800", sft_data), (args.dpo_name, dpo_data)]:
    errors = [d for d in data.values() if not d["is_correct"] and "full_response" in d]
    if not errors:
        print(f"  {name}: 无 full_response 字段，跳过分类")
        continue
    cats = defaultdict(int)
    for d in errors:
        cats[classify_error(d)] += 1
    total_err = len(errors)
    print(f"  {name} ({total_err} 错误):")
    for cat in ["format_error", "no_answer", "close_miss", "reasoning_failure"]:
        print(f"    {cat:<20}: {cats[cat]:3d} ({cats[cat]/total_err*100:.1f}%)")

# ── 3. 难度分析（答案大小分组）──────────────────────────────
print(f"\n── 3. 难度分析（按答案大小分组）──")
for name, data in [("SFT-800", sft_data), (args.dpo_name, dpo_data)]:
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for d in data.values():
        b = answer_bucket(d["gold_val"])
        buckets[b]["total"] += 1
        if d["is_correct"]:
            buckets[b]["correct"] += 1
    print(f"  {name}:")
    for b in ["0-10", "11-100", "101-1000", "1000+"]:
        bd = buckets[b]
        if bd["total"] > 0:
            acc = bd["correct"] / bd["total"] * 100
            print(f"    答案 {b:<10}: {acc:.1f}% ({bd['correct']}/{bd['total']})")

# ── 4. 回答长度分布 ──────────────────────────────────────────
print(f"\n── 4. 回答长度（词数）──")
for name, data in [("SFT-800", sft_data), (args.dpo_name, dpo_data)]:
    lengths = [response_length(d) for d in data.values() if d.get("full_response")]
    if not lengths:
        print(f"  {name}: 无 full_response")
        continue
    avg = sum(lengths) / len(lengths)
    lengths_sorted = sorted(lengths)
    median = lengths_sorted[len(lengths_sorted)//2]
    print(f"  {name}: 均值={avg:.0f}词 中位数={median}词 最短={min(lengths)} 最长={max(lengths)}")

# ── 5. McNemar's test ────────────────────────────────────────
print(f"\n── 5. McNemar's test (SFT-800 vs {args.dpo_name}) ──")
b = len(sft_wrong_dpo_right)  # SFT错DPO对
c = len(sft_right_dpo_wrong)  # SFT对DPO错
print(f"  b(SFT错DPO对)={b}, c(SFT对DPO错)={c}")
if b + c == 0:
    print("  无法计算（无差异）")
else:
    # 使用连续性校正
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    print(f"  McNemar chi²(连续性校正) = {chi2:.3f}")
    if chi2 > 10.83:
        print("  p < 0.001 *** 极显著")
    elif chi2 > 6.63:
        print("  p < 0.01 ** 显著")
    elif chi2 > 3.84:
        print("  p < 0.05 * 显著")
    else:
        print("  p >= 0.05 不显著（差异在统计误差范围内）")

print(f"\n{'='*60}")
