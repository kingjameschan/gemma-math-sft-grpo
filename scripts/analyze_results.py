import json
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.abspath(os.path.join(script_dir, "../outputs"))

def analyze_jsonl(filepath, name):
    if not os.path.exists(filepath):
        return None
        
    correct = 0
    total = 0
    format_adhered = 0
    total_steps = 0
    
    calc_errors = 0
    logic_errors = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            total += 1
            if record['is_correct']:
                correct += 1
                
            response = record['full_response']
            
            # 格式遵循率：是否输出了 ####
            if "####" in response:
                format_adhered += 1
                
            # 平均推理步数：按换行符估算
            steps = response.count('\n')
            total_steps += steps
            
            # 简单错题归类 (基于试探性的规则)
            if not record['is_correct']:
                if "####" in response:
                    # 如果格式对了但答案错了，粗略判定为计算或逻辑错误
                    # 更深度的可以对比金标准里的中间数字，这里简化处理
                    calc_errors += 1
                else:
                    # 格式没对齐，大概率是还没学会 SFT 模板
                    logic_errors += 1

    return {
        "name": name,
        "acc": correct / total if total else 0,
        "format_rate": format_adhered / total if total else 0,
        "avg_steps": total_steps / total if total else 0,
        "calc_errors": calc_errors,
        "format_errors": logic_errors
    }

def main():
    print("="*60)
    print("📊 深度多维评估报告 (Advanced Metrics)")
    print("="*60)
    
    results = []
    
    # 按照训练步骤排序
    files_to_check = [
        ("Step 200", "eval_details_200.jsonl"),
        ("Step 400", "eval_details_400.jsonl"),
        ("Step 600", "eval_details_600.jsonl"),
        ("Step 800", "eval_details_800.jsonl"),
        ("Final (1 Epoch)", "sft_eval_results.jsonl")
    ]
    
    for name, filename in files_to_check:
        filepath = os.path.join(outputs_dir, filename)
        stats = analyze_jsonl(filepath, name)
        if stats:
            results.append(stats)
            
    # 打印格式化表格
    print(f"{ 'Checkpoint':<18} | {'Accuracy':<10} | {'Format Rate':<12} | {'Avg Steps':<10} | {'Format Errors'}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<18} | {r['acc']:<10.2%} | {r['format_rate']:<12.2%} | {r['avg_steps']:<10.1f} | {r['format_errors']}")
        
if __name__ == "__main__":
    main()
