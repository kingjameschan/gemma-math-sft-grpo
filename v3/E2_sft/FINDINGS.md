# E2 SFT 行为分析 — lr=5e-4 anchor 自顶向下

**起点问题**：SFT 后 pass@K 几乎不变、maj@K 跌 5pp、pass@1 跌 18pp — 三个量为何各自不变或变化？

**总览图**：`outputs/analysis_flowchart.png` — 三列分别追踪 pass@K / maj@K / pass@1 的解释路径，每条路径标注由哪一层（L2.x / L3 / L4 / L5）回答，底部 SYNTHESIS 给出综合结论。

**目的**：以 lr=5e-4 (r=64, 2 epoch, 186 steps, 10 ckpts × K=64 deep eval) 为锚点，自顶向下解构 SFT 在 Gemma2-2B-IT GSM8K 上的行为。

**5 层结构**：
- **L1** 宏观曲线 — pass@1 / pass@K / maj@K
- **L2** 分桶轨迹 — base 桶口径下 E/M/H 的演化（pass@1 mass + maj@K mode 双视角）
- **L3** 每题分布 — self 桶口径下 landscape 重塑 + 题级搬家
- **L4** 题内 K 分布形状 — mode_mass / wrong_concentration（区分 attractor vs dilution）
- **L5** response 物理形态 — token 长度 / step 数（解释 L1-L4 的物理机制）

**两条横向对照**：vs base IT (E1)；vs lr=1e-4 / 1e-3。

## 主集合图（step 130 vs base 全 5 层）

`outputs/lr5e-4_step130_combined.png` — 5×2 = 10 panel，把 L1-L5 所有引用的 step 130 vs base panel 收成一张。

| Panel | 内容 | 来源 |
|---|---|---|
| L1.1 | pass@K + maj@K curves | `passk_majk_curves.png` 中列 / `passk_majk_by_ckpt.png` step 130 panel |
| L1.2 | **ABC decomposition stacked bar** | **NEW** — 此前无图 |
| L2.1 | per-bucket pass@1 | `sft_per_bucket_trajectory.png` 中列下排（step 130 取点） |
| L2.2 | **per-bucket maj@K + flip 计数** | **NEW** — 此前无图 |
| L3.1 | per-Q pass@K 分布 base vs SFT 叠图 | `lr5e-4_difficulty_grid.png` step 130 panel + E1 base `base_gemma-2-2b-it_k64_difficulty_buckets.png` |
| L3.2 | **base→SFT 题级搬家 heatmap** | **NEW** — 此前无图 |
| L4.1 | mode_mass vs correct_mass scatter | `lr5e-4_mode_correctness_scatter.png` base panel + step 130 panel |
| L4.2 | wrong_concentration 直方图 | `lr5e-4_wrong_concentration.png` 左 panel（step 130 是 best 之一） |
| L5.1 | response token 长度直方图 | `length_e1style/lr5e-4_step130_length_e1style.png` A.1 + base `length_classA.png` A.1 |
| L5.2 | step 数直方图 | `length_e1style/lr5e-4_step130_length_e1style.png` A.5 marginal |

**轨迹（多 ckpt）视图仍看原 detail 图**（每 panel 引用时带 ckpt 范围）：
- 多 ckpt pass@K/maj@K 演化：`passk_majk_curves.png` / `passk_majk_by_ckpt.png`
- 多 ckpt 桶轨迹：`sft_per_bucket_trajectory.png`
- 多 ckpt self landscape：`lr5e-4_difficulty_grid.png`
- 多 ckpt mode/wrong：`lr5e-4_mode_correctness_scatter.png` / `lr5e-4_wrong_concentration.png` 右 trajectory
- 多 ckpt 长度：`lr5e-4_length_grid.png` + `length_e1style/*.png` × 10

每个发现旁边写 "看 `<文件>` 哪个 panel 的什么颜色/位置"。step 130 snapshot 优先指 combined 图，trajectory 优先指对应 detail 图。

**3 个 LR 都已渲染**：`lr1e-4_step186_combined.png`、`lr5e-4_step130_combined.png`、`lr1e-3_step110_combined.png`（best ckpt 各自）。

---

## 6 大行为概括（执行摘要）

每条都标注**支持的图 + panel + 视觉特征**，方便快速 trace。

### **1. Capability 没丢，采样效率大伤** (L1)

**图**：3 张 combined 各自的 **L1.1**（pass@K + maj@K 曲线）+ **L1.2**（ABC 堆叠柱）

**核心**：
- L1.1：4 条线**右端贴 + 左端散** = pass@K 保留 + pass@1/maj@K 跌
- L1.2：A 段（mode=correct）跌 6.1pp、B 段（会但 mode 错）涨 6.1pp、**C 段（完全不会）几乎不变**

→ SFT 不是删除能力，是采样错位

### **2. 不对称冲击：重伤 Easy，微帮 Hard** (L2，base 桶口径)

**图**：3 张 combined 各自的 **L2.1**（per-bucket pass@1）+ **L2.2**（per-bucket maj@64 + flip 计数）

**核心**：
- Easy −27pp pass@1（贡献 60% 总损失）
- Hard +5pp pass@1（dilution 副作用）
- L2.2 Hard maj@64 base=**0%** → SFT=12%（结构性单向涨，非真学）

→ SFT 优先击垮原本最稳定的题

### **3. Landscape 镜像翻转 + 题级单向流出** (L3，self 桶口径)

**图**：3 张 combined 各自的 **L3.1a/b**（per-Q pass@K 分布拆图）+ **L3.2**（3×3 migration heatmap）

**核心**：
- L3.1：base 右偏 J 形（右尾大 spike 440 题完美）→ SFT **镜像翻转**到左偏 J 形（左尾大 spike 250 题几乎全错）；**median pass@K 0.78 → 0.375（−40pp）**
- L3.2 migration matrix：**60% base Easy 单向流到 Medium**（仅 4% 反向回补）；E↔H 跨两级极少（<3%）

→ 不是少数题崩，是**中位数题位置整体左移**；Easy 流出几乎不可逆

### **4. Distribution Dilution（不是 wrong attractor）** (L4)

**图**：3 张 combined 各自的 **L4.1a/b**（mode_mass vs correct_mass KDE 云图）+ **L4.2**（wrong_concentration 直方图）

**核心**：
- L4.1：base (1, 1) 角紧凑团 + 左上强 wrong attractor 团（13 题）→ SFT **两团都被打散**；云沿 y=x **滑向中段**
- L4.2：base 直方图右尾 spike 集中 → SFT **左移到中段**
- mode_mass 跌 0.21 + wrong_conc 跌 0.22 = 同步下降

→ **几何运动 = 沿 y=x 滑（dilution）**；wrong attractor 假设被排除

### **5. 物理机制 = "格式化短答"** (L5)

**图**：3 张 combined 各自的 **L5.1**（response token 长度直方图）+ **L5.2**（step marker 数直方图）

**核心**：
- L5.1：mean response token **192 → 137（−29%）**；分布主峰左移 + 收窄
- L5.2：含 ≥1 step marker 的 response 比例 **+20pp**（25% → 45%）

→ "**短 token + 多 numbered step** = 每步内容变薄" = NLL 推动的副作用

### **6. LR-Robust Attractor + Damage Threshold** (3 LR 横向对比)

**图**：**3 张 combined 图并排**比较，重点看：
- 标题里 Δ 数字（per-LR 自动计算）
- **L1.1**：1e-4/5e-4 K=64 处贴 base；**1e-3 全 K 段都低**
- **L1.2**：C 段 1e-4(6.6%) ≈ 5e-4(6.7%) ≈ base；**1e-3 = 11.4%**
- **L4.1**：1e-4/5e-4 云形几乎一致；**1e-3 (1,1) 角更稀、左上 attractor 更密**

| 量 | 1e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| pass@K=64 vs base | +0.01 | 0 | **−0.04** |
| C 段 % | 6.6 | 6.7 | **11.4** |
| mode_mass | 0.50 | 0.51 | 0.47 |

→ **1e-4 ≈ 5e-4 收敛于 dilution attractor**；**1e-3 跨过 capability damage 阈值**

### **核心 takeaway（一句话）**

> SFT 在 6 个独立维度上**全部指向同一机制：distribution dilution = anti-RL**。
>
> RL 把 correct 推到 top-1 (DSMath §"Why RL Works")；
> **SFT 把 correct 推回 top-K + wrong 也散乱**，是镜像反向操作。
>
> **最简单的视觉证据**：3 张 combined 图的 **L1.1 panel** —— 4 条曲线的"右端贴 + 左端散" + "maj@K 永远在 base 之下" = SFT 的全部宏观行为画像。

---

## 层间关系（dependency graph）

每层粒度递减，每层回答上层的留白，并新提一个更细的问题给下层。

```
                                   粒度
L1  Macro       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1 数/ckpt
                ↓ 留白: 5% mode 由对变错——是哪 5%？
                ↓
L2  Buckets     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3 数/ckpt (base 桶)
                ↓ 留白: Easy 崩 27pp——是 capability lost 还是 sampling drift？
                ↓
L3  Per-Q       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1319 数/ckpt + 搬家矩阵
                ↓ 留白: Easy 题丢的 17 个 correct sample——错答案集中 vs 散乱？
                ↓
L4  K-Shape     ━━━━━━━━━━━━━━━━━━━━━━━ 1319 × {mode_mass, wrong_conc} /ckpt
                ↓ 留白: 分布形状变化对应 response 什么物理变化？
                ↓
L5  Physical    ━━━━━━━━━━━━━━ 1319 × 64 ≈ 84k responses 的 token/step
                ↺ 闭环回 L1: 物理变化解释 pass@1 为何崩
```

### 每层的输入/输出/留白

| 层 | INPUT (上层留白) | OUTPUT (本层结论) | NEXT LAYER 留白 |
|---|---|---|---|
| **L1** | (无) | pass@1 砍 1/3 + maj@K 救不回；ABC: 5% 题 mode 由对变错 | "**哪些题** mode 变错了？是按难度集中的吗？" → L2 |
| **L2** | L1 的"哪些题"问题 | Easy 崩 27pp，Medium 崩 19pp，Hard 升 5pp；60% 损失来自 Easy | "Easy 崩是真 capability 丢了，还是只是 K=1 sampling 漂移？" → L3 |
| **L3** | L2 的 capability vs sampling 问题 | capability 真丢：Easy 桶 pass@K 0.98→0.71；60% Easy 单向流到 Medium | "Easy 题丢的 17 个 correct sample，**错答案是集中在一个还是散乱**？" → L4 |
| **L4** | L3 的 wrong-mass 去向问题（待） | mode_mass + wrong_concentration 是否下降 → 区分 attractor vs dilution | "K 分布形状的变化对应 response 什么**物理变化**（长度/step）？" → L5 |
| **L5** | L4 的 shape→physical 映射问题（待） | response token 长度 / step 数 / 推理深度的演化 | "物理变化为何能解释 L1 的 pass@1 崩"——闭环回 L1 |

### 横向对照贯穿 5 层

每层都有两条对照线：
- **vs base IT (E1)**：SFT 偏离 base 多少
- **vs lr=1e-4 / 1e-3**：5e-4 在 LR 谱系中的相对位置

### 不是单线关系——L1 有两条留白

L1 实际上提了两个不同问题，分给两层：
- **"哪些题"** → **L2** 按难度桶切
- **"mode 错时是真 attractor 还是 dilution"** → **L4** 题内 K 分布形状

L2 → L3 是单线（capability vs sampling），L3 → L4 也是单线（wrong-mass 去向）。

---

## L1 — 宏观曲线

**用图**：
- `outputs/passk_majk_curves.png`（中列 = lr=5e-4）
- `outputs/passk_majk_by_ckpt.png`（绿线 = lr=5e-4，每 panel 黑线 = base IT 参考）

### 3 个核心事实

1. **能力 ceiling 几乎保留**：pass@64 base 93% → SFT 88-93%（gap ≤5pp）
2. **pass@1 砍 1/3**：base 62% → SFT 30-45%（best step130 = 44%，仍 −18pp）
3. **maj@K 救不回**：base 70% → SFT 50-65%（best 65%，仍 −5pp）；voting gain (maj@K − pass@1) base=8pp，SFT=21pp → **SFT 样本波动更大**

### ABC decomposition（K=64 题级状态）

精确数（精确到 0.1pp）：

| 区域 | 含义 | base | SFT 130 | Δ |
|---|---|---|---|---|
| A | mode = correct | **69.8%** | **63.7%** | **−6.1pp** |
| B | 有对答案但 mode 错 | 23.5% | 29.6% | **+6.1pp** |
| C | 0 个样本对 | 6.7% | 6.7% | 0 |

→ **6.1% 题从 A 搬到 B**：mode 由对变错，但能力本身没失传。
→ ΔA = −6.1pp 即 macro maj@K 跌幅，与 L1.1 数一致。

### C 段 churn — count 不变 ≠ 集合不变

ΔC ≈ 0（base 88 → SFT 89）但**集合换了一半**：

| 类别 | 题数 | % |
|---|---|---|
| both C（持续不会）| **44** | 3.3% |
| only base C（SFT 救回）| 44 | 3.3% |
| only SFT C（SFT 新整丢）| 45 | 3.4% |
| **总 churn** | **89** | **6.7%** |

**救回的 44 题**（base C → SFT non-C）：SFT mean c=7.3/64，**0 题升到 Easy**，最多到 Hard+/Medium- 边缘。

**新整丢的 45 题**（base non-C → SFT C）：base mean c=6.4，**0 题原本是 Easy**，全是 Medium 边缘 (15) + Hard 边缘 (30)。

→ **C 段 churn 集中在 Hard 边缘的 oscillation**：base 与 SFT 在 c≈0-7 区间互换名单，但**Easy 永远不会跌进 C，C 也永远不会跳到 Easy**。

→ ΔC=0 看似稳定，实际隐藏 6.7% 的 swap，真正"持续不会"只有 3.3% (44 题)。

### 跨 ckpt 趋势

10 个 ckpt 在 pass@K 曲线上**几乎纠缠成一束**，没有单调趋势。step 10 的 pass@1=30% 是最差，step 130=44% 最好，**没有任何 ckpt 在 pass@1 / maj@K 上接近 base**。

### L1 边界

- 能说："mode ≠ correct 的题多了 5pp"
- **不能说**："mode 是真集中到错答案 vs mass 散乱"——这两种情形 maj@K 都看不出，需 L4 的 mode_mass / wrong_concentration 区分

---

## L2 — 分桶轨迹（base 口径）

**用图**：`outputs/sft_per_bucket_trajectory.png` 中列下排（实线 SFT，虚线 base 同色参考）

**桶口径**：base IT pass@K=64 切的 fixed bucket（E≥0.9 / M / H≤0.1），label 文件 `v3/shared/data/gsm8k/test_difficulty_labels.jsonl`，1319 题永远归属同一个桶。

### 桶定义 + base baseline

| Bucket | 题数 | 基线 base avg pass@1 (stoch) |
|---|---|---|
| Easy (517) | 39% | **98.0%** |
| Medium (562) | 43% | **52.8%** |
| Hard (240) | 18% | **2.9%** |

### 3 个非对称变化（best ckpt step=130）

| Bucket | base | SFT 130 | Δ | 对总 pass@1 贡献 |
|---|---|---|---|---|
| Easy (517) | 98% | 71% | **−27pp** | **−10.5pp** |
| Medium (562) | 53% | 34% | **−19pp** | **−7.9pp** |
| Hard (240) | 3% | 8% | **+5pp** | +0.9pp |
| 合计 | 62% | 44% | **−17.6pp** | ✓ 与 L1 macro 对上 |

### 视觉证据（图中位置）

下排 panel 是**唯一红实线超过红虚线**的位置；绿/橙实线均显著低于各自虚线。

### 关键观察

1. **Easy 桶贡献 60% 损失** — SFT 主要伤"原本最稳定"的题
2. **Hard 桶 +5pp 仅 ~3% 是 format gain**（统计 29 道 Hard mode-flip→C 题，仅 1 道是单位/格式问题；97% 是真 mode shift）；但**SFT mode freq 多数较弱（12-30%），是分布稀释让 correct 微弱多数当上 mode**，不是真"学会"
3. **trajectory 全程在 base 远下方**，best ckpt 也救不回 Easy

### L2.2 — per-bucket maj@64（mode 视角）+ flip 计数

base 桶口径下，比较 base 与 SFT 的 maj@64：

| Bucket | size | base maj@64 | SFT maj@64 | Δ | flip→W | flip→C |
|---|---|---|---|---|---|---|
| Easy | 517 | **99.6%** | 89.6% | **−10pp** | **54** | 2 |
| Medium | 562 | 72.2% | 61.9% | **−10pp** | **128** | 70 |
| Hard | 240 | **0.0%** | **12.1%** | **+12pp** | 0 | **29** |
| **TOTAL** | 1319 | 69.8% | 63.7% | **−6.1pp** | 182 | 101 |

→ 净 flip = 182 − 101 = 81 = 6.1pp ✓ 与 L1 ABC ΔA 对上。

**flip 中的 mode-shift 类型分类**（共 1319 题里所有 flip 题）：

| 类型 | 计数 | % |
|---|---|---|
| **(I) Format gain**（单位/格式差）| 4 | 0.3% |
| **(II) Close-miss shift**（数值接近 gold 但不等）| 86 + 43 = 129 | 9.8% |
| **(III) Far-shift**（数值与 gold 完全不同）| 96 + 54 = 150 | 11.4% |

→ **Format gain 可以忽略（0.3%）**，绝大多数 flip 是真 mode shift。

**Easy 桶 flip→W 异常**：54 题中 31 题（57%）是 III far-shift（SFT 在原本 base 完美的题上输出了 ×3 / ÷2 等结构性新错答案）。这意味着 SFT **主动产生新的错误算法路径**，不是简单"忘记"。

### 两条 L2 故事（mass vs mode 视角）

| 视角 | Easy | Medium | Hard |
|---|---|---|---|
| **L2.1 pass@1（mass）** | **−27pp** dominant | −19pp | +5pp |
| **L2.2 maj@64（mode）** | −10pp | −10pp | **+12pp** |

→ mass 损失主要砸 Easy；mode 翻车 Easy/Med 平摊，Hard 反获益（dilution 副作用让弱 correct 当上 mode）。

### L2 边界

Easy 桶崩 27pp，是因为：
- (a) 这些题 pass@K=64 也跟着崩了（capability lost）
- (b) pass@K 还在 base 水平，但 K=1 sampling 不再选对（sampling drift）

→ L3 解决。

---

## L3 — 每题分布重塑 + 搬家矩阵

**用图**：
- `outputs/lr5e-4_difficulty_grid.png` step=130 panel
- 对照 `v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64_difficulty_buckets.png`
- **搬家矩阵无现成图**（脚本算的，未来可补一张可视化）

**重要口径区分**（与 L2 对比）：

| | L2 图 | L3 图 |
|---|---|---|
| bucket 来源 | base IT pass@K 切的 fixed bucket | 每个 ckpt 自己 pass@K 切的 self bucket |
| 同一道题不同 ckpt | 永远同一桶 | 不同 ckpt 可能不同桶 |
| 用途 | like-for-like capability shift | self landscape reshape |

### landscape 整体左移（self 桶口径）

|  | base IT | SFT step 130 |
|---|---|---|
| Easy 桶 (≥0.9) | **517** | **212** (−59%) |
| Medium 桶 | 562 | 774 (+38%) |
| Hard 桶 (≤0.1) | **240** | **333** (+39%) |
| 全分布 mean pass@K | 0.614 | **0.439** |
| 全分布 median pass@K | **0.781** | **0.375** |

→ median 从 0.78 砍到 0.375，**整个分布左移 40pp** — 不是少数题崩，是中位数题崩。

### Base → SFT step 130 题级搬家矩阵（1319 题）

```
              → SFT E   → SFT M   → SFT H   |  hold rate
base E (517)    190       311       16      |  37%   (60% to M, 3% to H)
base M (562)     20       412      130      |  73%   ( 4% to E, 23% to H)
base H (240)      2        51      187      |  78%   ( 1% to E, 21% to M)
```

**两条边界不对称**：
- **E↔M 边界单向流出**：60% Easy→Medium，仅 4% Medium→Easy（净 −291）
- **M↔H 边界对称流动**：23% Medium→Hard 对 21% Hard→Medium（净 −79）
- **Hard 几乎不会被打回 Easy**（仅 2/240 = <1%）

### capability 真实下沉（直接回答 L2 留白）

base 桶口径下，每桶平均 pass@K：

| base 桶 | base avg pass@K | SFT 130 同桶 avg pass@K | Δ |
|---|---|---|---|
| Easy (517) | 0.98 | **0.71** | **−0.27** |
| Medium (562) | 0.53 | 0.34 | −0.19 |
| Hard (240) | 0.03 | 0.08 | +0.05 |

→ Easy 桶**不是 sampling 漂移**：64 个样本里 base 平均答对 63 个，SFT 130 平均只对 46 个 — **真丢了 17 个 correct mass**。

### L3 边界

Easy 桶丢的 17 个 correct 样本，**重新分配到了哪些错答案**？
- 集中到一个具体的错答案（wrong attractor）？
- 还是散乱分布（diluted）？

→ L4 的 mode_mass + wrong_concentration 给出。

---

---

## L4 — 题内 K 分布形状

**用图**：
- 集合图 `outputs/lr5e-4_step130_combined.png` L4.1 + L4.2 panels（step 130 vs base）
- 多 ckpt 轨迹见 `outputs/lr5e-4_mode_correctness_scatter.png`（3×4 grid）+ `outputs/lr5e-4_wrong_concentration.png`

### 概念定义

每题 64 sample 的两个分布形状指标：

**`mode_mass`** = max{freq(y) : y ∈ 答案集合} / 64 = "mode 答案的频率占 K 的比例"
- 1.0 = 64 sample 全输出同一答案（极强 mode）
- ~0.05 = mode 仅微弱多数（散乱 dilution）

**`wrong_concentration`** = `top_wrong_freq / total_wrong_freq`，仅对有 ≥1 错样本（c < 64）的题计算
- `top_wrong_freq` = 频率最高的错答案的样本数
- `total_wrong_freq` = 64 − c
- 1.0 = 所有错样本都是同一个错答案（强 wrong attractor）
- ~0.05 = 错样本散到多个不同错答案（dilution）

### 几何约束（L4.1 散点图必须满足）

每题 (correct_mass, mode_mass) 落在以下区域：
- y = x（mode = correct，沿对角线）
- 或 y > x AND y + x ≤ 1（mode 是某个错答案，与 correct 共享 K=64 sample）
- 其他区域几何不可能

→ 当 correct_mass > 0.5，mode 必是 correct（correct 自己已占 ≥50%），点必在 y=x 上。

### 关键读数

| 指标 | base | SFT 130 | Δ |
|---|---|---|---|
| **mean mode_mass** (L4.1) | **0.72** | **0.51** | **−0.21** |
| **mean wrong_concentration** (L4.2) | **0.59** | **0.37** | **−0.22** |
| % mode==correct (= maj@64) | 70% | 64% | −6pp |
| 极端 wrong attractor 题数（c=0, mode_mass≥0.95）| **13** | **1** | −12 |

→ **mode 端跌 0.21 + wrong 端跌 0.22**，幅度一致。

### 区分 attractor vs dilution（联合证据）

| 假设 | 预期 mode_mass | 预期 wrong_conc | 与观测匹配 |
|---|---|---|---|
| (a) Wrong attractor（mass 全搬到一个错）| 涨 | 涨 | ❌ 都跌 |
| (b) Distribution dilution（mass 散乱）| 跌 | 跌 | ✓ 完美匹配 |
| (c) Mode 翻 + wrong 集中 | 涨 | 跌 | ❌ mode 没涨 |

→ **只有 (b) Distribution dilution 同时解释 L4.1 + L4.2 的两个跌幅**。

### 几何关键发现 — "全方位 dilution"

L4.1 KDE 云图显示：
- **base 在 (1, 1) 角** 有强 correct mode 团（Easy 题集中）
- **base 在左上角** 有 ~70 道强 wrong attractor 团（Hard 题，mode 错但强）
- SFT 把**两个团都打散**：
  - (1, 1) 团 → 滑向中段对角线
  - 左上 attractor 团 → 也散乱（69 → 18 道）

→ SFT 的 dilution 是**对称的**：不论 mode 是对是错，**任何高 confidence 都被稀释**。

### L4 → L5 留白

L4 揭示分布形状变化，但没回答"**为什么会 dilute**"——是 model 输出了不同长度的 response？还是不同推理结构？还是别的物理变化？

→ L5 看 response token 长度 + step count 给物理解释。

---

## L5 — response 物理形态

**用图**：
- 集合图 `outputs/lr5e-4_step130_combined.png` L5.1 + L5.2 panels
- 多 ckpt 长度 trajectory：`outputs/lr5e-4_length_grid.png`（chars）+ `outputs/length_e1style/*.png` × 10（token + E1 同构）
- base 参考：`v3/E1_baseline/outputs/pass_at_k_20260427_222954/base_gemma-2-2b-it_k64_length_classA.png`

### 概念定义

每个 response = 模型对一道题的一次完整输出。1319 题 × 64 sample = **84,416 个 response** 用于 L5 统计。

**`response token length`** = 该 response 用 Gemma2-IT BPE tokenizer 编码后的 token 数（不含 prompt）

**`step count`** = response 文本中正则 `**N.` 或 `**Step N:` 的匹配数。这测的是"模型有没有用 numbered list 格式排版"，**不直接等于"推理步骤数"**——只是格式化代理指标。

### 关键读数

| 指标 | base | SFT 130 | Δ |
|---|---|---|---|
| **mean response tok** (L5.1) | **192** | **137** | **−55 tok (−29%)** |
| p99 response tok | 426 | 389 | −37 |
| mode (peak bin) tok | 158-175 | 88-105 | mode 左移 ~70 tok |
| **mean step count** (L5.2) | ~0.2 | ~0.5 | +0.3 |
| % response 含 ≥1 step marker | ~25% | ~45% | **+20pp** |

### 反直觉发现 — "短 + 步骤化"

L5.1 token 变短 + L5.2 step marker 变多 = **特殊的"格式化短答"模式**：

| 模型 | 典型 response 形态 |
|---|---|
| base | 自由叙述 + 长推理 + 答案。"Janet sells 16-3-4 = 9 eggs. She makes 9*2 = $18 daily. \boxed{18}"（无 numbered step，~150-200 tok）|
| SFT | numbered short steps。"**1.** Compute eggs sold: 9. **2.** Multiply: 18. \boxed{18}"（有 step marker，但每 step 内容更薄，总长度 ~100-130 tok）|

→ SFT 学到 gold 数据中部分 completion 的 "**N.**" 格式，**用更少 token 凑出更多 step marker** = 每 step 内容变薄。

### 物理机制 — 解释 L1-L4 的所有损失

| L1-L4 现象 | L5 物理对应 |
|---|---|
| pass@1 跌 18pp | 短答 → K=1 抽样命中 correct 概率减半 |
| mode_mass 跌 0.21 | 短答给的"内容差异"小 → 多个不同短答各占类似份额 → mode 弱 |
| wrong_conc 跌 0.22 | 同上，wrong 也散到多个短答 |
| Easy −27pp | base "长推理优势" 被砍 |
| Hard +5pp | Hard 题 base 长推理也错 → 短答没什么可丢 |

→ L5 是 L1-L4 现象的**根源**：SFT 把模型从"长推理 + 强 mode"推向"短答 + 弱 mode + 多样性散乱"。

### 两个推测的 SFT 内在机制

**(1) NLL 推动早出 \boxed{}**：gold 的 \boxed{} 出现在末尾，SFT 学"快点结束 → boxed → 拿低 NLL"，副作用是推理被截短。

**(2) Gold 的格式与 base 的自由格式 mismatch**：gold 用 "**N.**" 数字步骤，base 自由文字。SFT 把 base 的高 mode 推开，朝 gold 格式靠拢，**但 token budget 被 step 标记吃掉**，每步内容变薄。

---

## 完整 L1 → L5 故事链

```
L1: pass@1 -18pp, maj@K -6pp, pass@K 不变
   ↓ (哪些题？)
L2.1: pass@1 损失主在 Easy (-27) + Medium (-19); Hard +5
L2.2: maj@K 损失 Easy/Med 平摊 (-10 each); Hard +12 (无对照可翻)
   ↓ (Easy 是真衰减还是 sampling drift?)
L3.1: 分布镜像翻转 (median 0.78 → 0.375), Easy 大量 (~290 题) 被推到中段
L3.2: 60% E→M 单向流出, M↔H 近对称, E↔H 极少
   → Easy 是真 mass attenuation (差 17 个 sample, 远超 K=64 抽样波动)
   ↓ (那 17 个 wrong sample 是集中还是散乱?)
L4.1: mode_mass 普遍跌 0.21；左上强 wrong attractor 团也散开
L4.2: wrong_conc 跌 0.22 与 mode_mass 一致
   → 全方位 distribution dilution（mode + wrong 都散），不是 attractor
   ↓ (物理上为什么 dilute?)
L5.1: response token 长度跌 29% (192 → 137)
L5.2: step marker 反而 +20pp = "格式化短答" 模式
   ↺ 闭环 L1: 短答 + 散乱 = pass@1 砍 1/3 + mode 弱 + wrong 散
```

---

## SYNTHESIS — SFT lr=5e-4 step 130 行为画像

> **SFT 没有"删除"任何题（pass@K 几乎不变, capability ceiling 保留）**，但通过物理上的"短答 + 格式化"机制，把模型从 base 的"长推理 + 强 mode" 状态推向"短答 + 弱 mode + 多样性散乱" 状态。
>
> **核心损失模式 = Distribution Dilution**：
> - 不是"忘记某些题"（capability lost）
> - 不是"换错答案"（wrong attractor）
> - 而是"所有强 confidence 被打平"（dilution）
>
> **Easy 桶受灾最重**（-27pp pass@1），因为 base 在 Easy 上的"强 correct mode"是 dilution 的主要稀释对象。
>
> **Hard 桶微涨**（+5pp pass@1, +12pp maj@K），但这不是 SFT 学到新能力，而是 dilution 副作用：base 的强 wrong attractor 散开后，correct 的微弱多数能踩着噪声地板成为新 mode。

---

## L1-L5 框架的可重用性（v3 protocol）

这套 5 层分析框架是**模型无关的**，对任何后续 RL 方法（RFT / online RFT / DPO / GRPO）都直接适用。每个 ckpt 只需：

1. K=64 sampling on D_test 1319 题
2. per-sample answers (用于 mode / wrong_conc)
3. per-sample response 文本（用于 length / step）

→ 跑完 RL 训练后，复用 `_plot_step130_combined.py`（改 ckpt 路径），自动生成 L1-L5 集合图。

**预期 RL 行为对照**（待 GRPO 验证）：

| RL 方法 | 预期 L4.1 mode_mass | 预期 L4.2 wrong_conc | 预期 L5 token |
|---|---|---|---|
| **DPO** | 涨（推 correct）| 可能涨 | 不确定 |
| **GRPO** | **大幅涨**（advantage sharpen 分布）| 可能涨 | 取决于 reward 是否惩罚长度 |
| **RFT** | 小幅涨（self-distill correct）| 可能涨 | 类似 SFT |

**最有价值的对照**：GRPO from base IT vs GRPO from SFT 130 → 直接验证"SFT 给 RL 留 diversity → RL 收益更大" 假设。

---

## 跨层开问题（部分已解决，部分留给 RL 阶段）

| 问题 | 状态 |
|---|---|
| ✓ Easy 题丢的 17 个 correct sample 是集中错 vs 散乱？ | L4 解决：是散乱（dilution）|
| ✓ 这种"重新分配"对应 response 什么物理变化？ | L5 解决：短答 + 格式化 |
| ✓ 为什么 SFT 反伤 Easy 不伤 Hard？ | L4 + L5：Easy 的强 correct mode 被 dilution 打散；Hard 的强 wrong attractor 也被打散反而带来微弱 +5 |
| 待 RL 验证：SFT 的 dilution 是否给 RL 留出空间？ | GRPO from base vs GRPO from SFT 130 对比 |
| 待 RL 验证：DPO/GRPO 能否 re-sharpen mode_mass？ | L4.1 trajectory tracking |
| 待 RL 验证：RL 后 response 长度是否回到 base 水平？ | L5.1 trajectory |
