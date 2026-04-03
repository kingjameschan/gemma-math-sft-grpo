# GCP Instance Instructions

你是一台 GCP L4 实例上的 Claude Code。请按照 `PLAN.md` 的步骤执行任务。

## 项目概述

Gemma2-2B-IT 数学推理微调项目。当前阶段：GRPO（在线强化学习）全量训练。
本地已完成 2000 样本 × 1 epoch 的 GRPO 试跑，验证了可行性。现在要在 L4 上用全量 7473 样本重跑。

## 关键指标

评测必须报告三个指标：
1. **####率**: 输出中包含 `####` 的比例
2. **数字正确**: fallback（最后一个数字）判断答案是否正确
3. **严格正确**（主指标）: `#### 纯数字`（不带单位）且数值正确

## 硬件

- GPU: NVIDIA L4 (24GB VRAM)
- 模型: Gemma2-2B-IT (~5GB bf16)
- 训练配置: bf16 + vLLM colocate 采样

## 模型信息

- Base model: `google/gemma-2-2b-it`（从 HuggingFace 下载）
- SFT adapter: `sft-adapter/`（从本 repo 的 `checkpoints/` 获取）
- Gemma2 不支持 system role，system prompt 折入 user message
- stop tokens: `["<end_of_turn>"]`

## 显存优化（重要）

训练和评测前，必须先探测最优参数以充分利用 L4 的 24GB VRAM：

**训练前**：
1. 用 `nvidia-smi` 确认可用显存
2. 逐步增大 `batch_size`（4→8→12）和 `num_generations`（4→8→12→16），找到不 OOM 的最大值
3. 调整 `vllm_gpu_util`（0.3→0.4→0.5），在训练和 vLLM 采样之间找最优平衡
4. 先用 `--num_samples 100` 小规模试跑 10 步验证不 OOM，再启动全量训练

**评测前**：
1. 调整 `gpu_memory_utilization`（默认 0.85，L4 可以试 0.90）
2. 确认 `max_model_len` 设置合理

**注意**：必须使用 `flash_attention_2`（已安装 flash-attn），不要降级到 SDPA。

## 执行步骤

请阅读 `PLAN.md` 并按顺序执行。每完成一步汇报结果。
