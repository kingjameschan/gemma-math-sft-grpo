# E5 — RL post-training with verl (DAPO + GRPO)

`gemma-2-2b-it` 上的 on-policy RLVR 实验与行为诊断。这里保留两条主线：

| Run | 方法 | 定位 |
|---|---|---|
| **R15 ck15** (`r15_dapo/`) | **DAPO** | Clip-Higher + Dynamic Sampling + Overlong Buffer + token-level loss |
| **R16 step42** (`r16_grpo_clean/`) | **GRPO** | group-relative advantage，group size 8，no critic |

R11–R14 是 verl 环境与 reward 链路的中间迭代，R17 是未完成的长跑；这些资产保留在本地，不作为公开仓库的 headline evidence。

## Canonical results

指标顺序：`pass@1 / pass@K / maj@K`。

| 方法 | GSM8K（n=1,319） | MATH-500-aug（n=500） |
|---|---:|---:|
| Base | K=128: 61.3 / 94.8 / 69.7% | K=128: 28.4 / 79.4 / 38.0% |
| DAPO R15 ck15 | K=64: **65.2 / 91.6 / 71.6%** | K=64: **31.0 / 73.7 / 40.0%** |
| GRPO R16 step42 | K=64: **66.6 / 92.3 / 73.4%** | K=128: **33.3 / 78.4 / 43.1%** |

- 在可比 K 上，DAPO / GRPO 提升 `pass@1` 与 `maj@K`，但 `pass@K` 与 Base 持平或略低。
- Base-anchored mode-mass 分析中，DAPO / GRPO 的正确主模态概率质量分别约 `+3.2 / +3.1 pp`，错误主模态约 `-7.8 / -16.1 pp`。
- 这些现象更符合对已有推理路径的概率重分配，而非当前有限 K 内可达题集的明确扩张。

> DAPO 和 GRPO 的步数、seed 与超参不完全对齐，因此 R16 只用作跨算法 robustness check，不作优劣排名。

## Evidence map

| 路径 | 内容 |
|---|---|
| [`outputs/k64_dapo_ck15/dapo_ck15_combined.png`](outputs/k64_dapo_ck15/dapo_ck15_combined.png) | DAPO 双数据集 L1–L10 dashboard |
| [`outputs/k64_r16_step42/r16_step42_combined.png`](outputs/k64_r16_step42/r16_step42_combined.png) | GRPO 双数据集 L1–L10 dashboard |
| [`outputs/yue_ppl_analysis/yue_8panel_selfppl.png`](outputs/yue_ppl_analysis/yue_8panel_selfppl.png) | Base / SFT / DAPO / GRPO / 外部链的同链 PPL probe |
| `outputs/mode_mass_delta/` | base-anchored mode-mass migration |
| `outputs/eval_log.jsonl` | 精简运行索引（旧集合/旧口径需结合 run id 解读） |

## Layout

| 路径 | 内容 |
|---|---|
| `train/` | 训练入口 |
| `r15_dapo/`, `r16_grpo_clean/` | 主运行 launch / reward / checkpoint 脚本 |
| `eval/` | dev checkpoint 选择与评测 |
| `audit/` | reward judge audit |
| `tools/` | pass@K / maj@K、迁移矩阵、mode mass 与 PPL 绘图 |
| `outputs/` | 精简日志与核心图表 |

## Environment

RL 运行于 verl + Ray + vLLM 环境，使用云端 L40S / L20。可复现环境见仓库根目录 [`docker/Dockerfile.grpo`](../../docker/Dockerfile.grpo) 与 [`requirements-grpo.txt`](../../requirements-grpo.txt)。
