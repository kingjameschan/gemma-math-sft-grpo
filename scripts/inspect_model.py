"""
检查模型内部结构。

用法：
  ~/vllm-env/bin/python scripts/inspect_model.py --model models/gemma-2-2b-it
  ~/vllm-env/bin/python scripts/inspect_model.py --model models/Qwen3-1.7B-Base
  ~/vllm-env/bin/python scripts/inspect_model.py --model models/Qwen3-8B-Base --layer 0
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def human_size(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    return f"{n/1e3:.1f}K"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None, help="打印第几层的详细结构（默认不打印）")
    parser.add_argument("--full", action="store_true", help="打印完整 print(model) 输出")
    args = parser.parse_args()

    # ── 加载 config（不加载权重，速度快）──────────────────────────────────────
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 60)
    print(f"模型: {args.model}")
    print("=" * 60)
    print(f"model_type:          {cfg.model_type}")
    print(f"num_hidden_layers:   {cfg.num_hidden_layers}")
    print(f"hidden_size:         {cfg.hidden_size}")
    print(f"num_attention_heads: {cfg.num_attention_heads}")
    print(f"num_key_value_heads: {getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}")
    print(f"head_dim:            {getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}")
    print(f"intermediate_size:   {getattr(cfg, 'intermediate_size', 'N/A')}")
    vocab_size = getattr(cfg, 'vocab_size', 'N/A')
    print(f"vocab_size:          {vocab_size}")
    print(f"max_position_embs:   {getattr(cfg, 'max_position_embeddings', 'N/A')}")
    rope_theta = getattr(cfg, 'rope_theta', None)
    rope_scaling = getattr(cfg, 'rope_scaling', None)
    print(f"rope_theta:          {rope_theta}")
    print(f"rope_scaling:        {rope_scaling}")
    print(f"hidden_act:          {getattr(cfg, 'hidden_act', 'N/A')}")
    print(f"attn_logit_softcap:  {getattr(cfg, 'attn_logit_softcapping', 'N/A')}  (Gemma2特有)")
    print(f"final_logit_softcap: {getattr(cfg, 'final_logit_softcapping', 'N/A')}  (Gemma2特有)")
    print(f"query_pre_attn_scalar: {getattr(cfg, 'query_pre_attn_scalar', 'N/A')}  (Gemma2 head_dim scale)")
    sliding_window = getattr(cfg, 'sliding_window', None)
    print(f"sliding_window:      {sliding_window}  (Gemma2 local attn window)")

    # ── 加载模型（cpu，bf16，快）──────────────────────────────────────────────
    print("\n加载模型权重（CPU bf16）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── 参数统计 ──────────────────────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 60}")
    print(f"总参数量:   {human_size(total)} ({total:,})")
    print(f"可训练参数: {human_size(trainable)} ({trainable:,})")

    # ── 各模块参数分布 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("模块参数分布（Top-level）：")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<30} {human_size(n):>8}  ({n/total*100:.1f}%)")

    # ── 单层详细结构 ──────────────────────────────────────────────────────────
    # 找 decoder layers
    layers = None
    for attr in ["layers", "h", "blocks"]:
        candidate = getattr(getattr(model, "model", model), attr, None)
        if candidate is not None:
            layers = candidate
            break

    if layers is not None:
        print(f"\n{'=' * 60}")
        print(f"共 {len(layers)} 个 decoder layer")
        layer_params = sum(p.numel() for p in layers[0].parameters())
        print(f"每层参数量: {human_size(layer_params)}")

        if args.layer is not None:
            idx = args.layer
            print(f"\n── Layer {idx} 详细结构 ──")
            layer = layers[idx]
            print(layer)
            print(f"\nLayer {idx} 各子模块参数量：")
            for name, mod in layer.named_modules():
                if name == "": continue
                n = sum(p.numel() for p in mod.parameters(recurse=False))
                if n > 0:
                    shapes = {pn: list(p.shape) for pn, p in mod.named_parameters(recurse=False)}
                    print(f"  {name:<45} {human_size(n):>8}  {shapes}")

    # ── 完整 print(model) ─────────────────────────────────────────────────────
    if args.full:
        print(f"\n{'=' * 60}")
        print("完整模型结构：")
        print(model)


if __name__ == "__main__":
    main()
