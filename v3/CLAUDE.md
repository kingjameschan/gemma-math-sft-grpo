# v3 canonical project context

> This file is contributor/agent context. The repository root `README.md` is the public source of truth.

## Goal

在小型 instruct 模型上分析 SFT、DAPO 与 GRPO 如何改变数学推理的输出分布，回答：RLVR 的增益更像能力边界扩张，还是对 Base 已有推理路径的概率重分配。

定位是 **empirical replication + behavioral diagnosis**，不声称新 RL 算法。

## Canonical scope

- Base: `gemma-2-2b-it`
- Evaluation: GSM8K test 1,319 + MATH-500-aug 500 numeric-verifiable questions
- Methods: Base, SFT (`lr=5e-4, ck130`), DAPO (`R15, ck15`), clean GRPO (`R16, step42`)
- Metrics: `pass@1 / pass@K / maj@K`, question-state transitions, difficulty buckets, answer-mode mass, same-chain PPL
- Answer checking: DeepSeek-style five-layer extraction + `math_equal`
- LoRA: `r=64, alpha=32, all-linear, dropout=0`

RFT / online RFT / DPO 属于历史规划，不再是当前 v3 canonical 实验主线。

## Canonical results

指标顺序：`pass@1 / pass@K / maj@K`。

| 方法 | GSM8K | MATH-500-aug |
|---|---:|---:|
| Base | K=128: 61.3 / 94.8 / 69.7% | K=128: 28.4 / 79.4 / 38.0% |
| SFT | K=128: 42.8 / 96.4 / 63.6% | K=128: 18.3 / 82.0 / 30.6% |
| DAPO | K=64: 65.2 / 91.6 / 71.6% | K=64: 31.0 / 73.7 / 40.0% |
| GRPO | K=64: 66.6 / 92.3 / 73.4% | K=128: 33.3 / 78.4 / 43.1% |

- MATH-500-aug 与旧 `math500_numeric` 293 题口径不可混用。
- 不同方法 Kmax 不同；只在共同 K 上比较曲线。
- DAPO 和 GRPO 的 step / seed / hyperparameters 不完全对齐；GRPO 是 robustness check，不作算法排名。

## Claim discipline

可以说：

- 当前实验域内，RLVR 提升 pass@1 / maj@K，但未观察到最大-K pass@K 的明确扩张。
- 题级迁移、mode mass 和 PPL probe 与“已有路径的概率重分配”一致。
- Base 对 RL 链的 PPL 接近 RL 模型自评，说明 RL 链仍在 Base 的高似然区域。

不可以说：

- RL 在任何模型/任务/训练预算下都不会提升能力。
- PPL 接近就证明 Base 能稳定采样出同样的正确链。
- GRPO 优于 DAPO，或 DAPO 优于 GRPO。

## Canonical layout

```text
v3/
├── E1_baseline/   # base + shared evaluation
├── E2_sft/        # SFT, checkpoint/LR sweep, behavioral findings
├── E5_grpo/       # verl DAPO R15 + clean GRPO R16
└── shared/        # shared extraction / equality utilities
```

## Environment

- Python 3.11 / CUDA 12.8 / torch 2.10
- SFT/training, vLLM evaluation and verl RL use separate dependency sets
- Local: RTX 5080 16 GB; RL: cloud L40S / L20
- See root `SETUP.md`, `docker/` and `requirements-*.txt`
