import json
import os
from datasets import Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "../data/gsm8k"))
output_dir = os.path.abspath(os.path.join(script_dir, "../data/sft_formatted"))

os.makedirs(output_dir, exist_ok=True)

# 统一的 System Prompt (由于 GSM8K 是英文数据集，这里使用英文以保持最佳性能)
SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Please reason step by step, and explicitly enclose your final numerical answer after '#### '."

def format_gsm8k_to_chatml(input_file, output_file):
    """
    将原始 jsonl {"question": "...", "answer": "..."}
    转换为标准 ChatML 格式的 messages 列表
    """
    formatted_data = []
    
    print(f"🔄 正在处理: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = item['question']
            answer = item['answer']
            
            # 构建对话 messages 列表
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            formatted_data.append({"messages": messages})
    
    # 将处理好的数据转换为 HuggingFace Dataset 并保存
    dataset = Dataset.from_list(formatted_data)
    
    # 存为 jsonl 格式方便抽查，也可以直接存为 parquet
    dataset.to_json(output_file, force_ascii=False)
    print(f"✅ 处理完成，共 {len(dataset)} 条数据。保存至: {output_file}\n")

if __name__ == "__main__":
    train_input = os.path.join(data_dir, "train.jsonl")
    test_input = os.path.join(data_dir, "test.jsonl")
    
    train_output = os.path.join(output_dir, "train_sft.jsonl")
    test_output = os.path.join(output_dir, "test_sft.jsonl")
    
    format_gsm8k_to_chatml(train_input, train_output)
    format_gsm8k_to_chatml(test_input, test_output)
    
    print("🎉 SFT 数据预处理全部完成！可以进入下一步训练了。")
