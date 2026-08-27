# TT-HN-OPD：动机、方法与最小验证计划

> 状态：研究设计草案，更新于 2026-08-27。本文把想法写成可证伪的实验计划，不把尚未跑出的结果写成结论。工作名 **TT-HN-OPD** 指 Trainable-Teacher Hard-Negative On-Policy Distillation，后续可以改名。

## 1. 一句话动机

GRPO 在同一个初始 prompt 下采样多条轨迹，因此能比较“同一个根状态下，哪些完整轨迹更好、哪些更坏”；但轨迹一旦分叉，后续状态几乎不再相同，终局 reward 只能以同一个粗粒度 advantage 广播到整条响应。OPD 让教师在学生真实访问的每个状态上提供 token 级方向，因此把监督从初始 prefix 延伸到了整条 on-policy 轨迹。我们进一步希望：**教师不只告诉学生应该靠近什么，还显式地从学生在同一状态产生的 hard negative 中学习应该远离什么。**

最关键的实验不是“学生会不会继续变强”——标准 OPD 已经大概率会让它变强——而是：

1. 可训练教师是否真的优于初始教师，而不只是熵变低、输出变尖；
2. 同状态学生负样本是否优于随机负样本或错配状态负样本；
3. 教师更新后，学生是否仍能稳定受益，而不是被一个漂移的教师带坏。

## 2. GRPO 的准确问题表述

给定 prompt x，GRPO 从当前策略采样 G 条轨迹：

τᵢ ∼ πθ(·∣x)，i = 1,…,G

第 i 条轨迹在时刻 t 的状态是：

sᵢ,ₜ = (x, aᵢ,<t)

组内 outcome reward 形成相对 advantage，例如：

Aᵢ = (Rᵢ − mean(R₁,…,R_G)) ÷ (std(R₁,…,R_G) + ε)

在常见 GRPO 实现中，Aᵢ 会广播到该响应的所有有效 token。需要特别澄清：**后续 token 不是没有梯度。** 每个 token 的 policy gradient 仍然以自己的完整历史 sᵢ,ₜ 为条件。真正缺少的是后续状态上的局部对照：

- t = 0 时，G 条轨迹确实共享同一个 x，因此存在同状态的多分支比较；
- 轨迹分叉后，sᵢ,ₜ 通常各不相同。即使两条轨迹后来碰巧出现同一个 token，它们的完整历史不同，也不是相同 prefix；
- 因而模型得到的是“这整条轨迹最终较好或较坏”，而不是“在这个具体中间状态，动作 a⁺ 比动作 a⁻ 好”。

### 2.1 问题一：局部反事实比较停留在根状态

终局 reward 可以给整条轨迹分配方向，却不能自动构造中间状态上的 sibling actions。它无法稳定回答：在已经写出这一段推理、已经执行这些工具、已经看到这些 observation 的条件下，下一步哪一个动作更好。

### 2.2 问题二：长程状态稀疏

长程 Agent 轨迹的状态空间随历史快速增长。某个精确状态 sₜ 往往只出现一次；模型可能只在这里见过一个成功动作，或只见过一个失败动作，没有足够的同状态正负样本建立决策边界。轨迹越长，这种稀疏越严重。

因此，更准确的说法不是“GRPO 不能监督后文”，而是：

> GRPO 有 state-conditioned gradient，但通常没有 state-local contrastive label；它在后续状态上缺少同条件的局部正负比较。

## 3. OPD 补上了什么

学生先从当前策略生成轨迹：

τS ∼ πS(·∣x)

在学生真实访问的每个状态 sₜˢ 上，冻结教师计算同一个学生 token aₜˢ 的概率。最常见的 sampled-token OPD 信号可以写成：

AOPD,ₜ = log πT(aₜˢ∣sₜˢ) − log πS(aₜˢ∣sₜˢ)

这样，教师和学生始终看到同一个 on-policy 状态 sₜˢ。教师无需等待这个精确状态在另一条轨迹中再次出现，就能立即告诉学生：当前 token 相对教师偏好是应提高还是应降低。

这正是 OPD 对长轨迹的价值：它不是让不同轨迹重新共享 prefix，而是在学生实际走到的每个状态上查询教师。

还要避免一个错误的新颖性表述：**标准 reverse-KL OPD 本来就可能给学生产生正、负两种 token 信号。** 我们的新增点不能只写成“引入负梯度”，而应写成：

> 保持教师作为学生的正向参照，同时让一个可训练教师显式排斥学生在同一 on-policy 状态产生的 hard negative；负梯度进入教师参数，而不只进入学生参数。

## 4. 提议方法：可训练教师 + 同状态 hard negative

### 4.1 两个角色、两条梯度路径

学生目标由 OPD 和可选 RLVR 构成：

LS = LRLVR + λOPD · LOPD

教师不接收 LS 的反向传播。教师有独立的 hard-negative preference loss：

LT = λHN · LHN + λanchor · DKL(πT ∥ πT₀)

其中 πT₀ 是初始教师的冻结副本，只用于限制教师漂移。两个优化器分开执行，不建立跨模型 autograd 图。

### 4.2 最小 token 级版本

对学生轨迹中的每个状态 sₜˢ：

- rejected token：aₜ⁻ = 学生实际采样的 aₜˢ；
- chosen token：aₜ⁺ = 更新前教师 πT,old 在同一状态的 top-1 token；
- 仅保留 aₜ⁺ ≠ aₜ⁻，且 aₜ⁻ 仍位于教师 top-k 内的分歧，避免把明显乱码当作“hard” negative；首轮设 k = 64，并做 k ∈ {16, 64, 256} 消融。

用冻结初始教师 πT₀ 构造 DPO 风格的相对 margin：

Δₜ = [log πT(aₜ⁺∣sₜˢ) − log πT(aₜ⁻∣sₜˢ)] − [log πT₀(aₜ⁺∣sₜˢ) − log πT₀(aₜ⁻∣sₜˢ)]

LHN = − meanₜ log σ(βΔₜ)

首轮使用 β = 0.1、λHN = 0.1、学生学习率 1×10⁻⁵、教师学习率 1×10⁻⁶。这些只是 pilot 起点，只能在训练集划出的 dev 上调整一次，不能看 AIME/HMMT test 调参。

这个版本几乎不增加生成成本：教师给 OPD 计算 logits 时，直接在设备内取 top-k 和两组 gather log-prob，随后丢弃完整 vocabulary logits。

### 4.3 一次 outer iteration

1. 冻结本轮行为快照 πS,old 与 πT,old。
2. 对每个 prompt 从 πS,old 采样 G 条轨迹，得到所有学生状态 sₜˢ。
3. πT,old 在相同状态上计算学生 token 的 OPD 分数、教师 top-k 和 aₜ⁺；所有监督张量 detach。
4. 更新学生：先做 OPD，实验需要时再加组内 RLVR advantage。
5. 从同一批状态构造 (sₜˢ, aₜ⁺, aₜ⁻)，用 LHN 更新教师；πT₀ 永远冻结。
6. 下一轮才使用更新后的学生和教师重新采样、重新打分。

先后顺序很重要：同一 batch 内给学生的教师分数必须来自教师更新之前，否则 supervision target 会在一个 optimizer step 内移动。

### 4.4 为什么无 verifier 也可能工作

无 verifier 版本依赖一个弱假设，而不是依赖“学生一定错”：

P(教师动作优于学生动作 ∣ 二者同状态分歧) > 1/2

只要教师总体更强，这个带噪 source prior 就可能在期望上产生正确方向。学生输出不必绝对错误；它只需平均弱于教师，就可以成为“相对负样本”。学生还会不断变化，因此会持续提供教师当前决策边界附近的新错误模式。

但这个机制也可能只是把教师原有偏好变得更尖，完全没有增加正确性。因此必须有以下对照：

- 同状态学生 negative；
- 将 negative 随机打乱到别的状态；
- 同状态随机 token negative；
- 只做教师 self-sharpening、不使用学生 negative；
- 更新教师但不更新学生。

如果这些对照与提议方法一样好，就不能声称“教师从学生错误中学到了东西”。

### 4.5 verifier 是可选校准器，不是方法成立的前提

第二阶段可以加入 outcome verifier：

- 学生失败：允许教师把学生分歧 token 当 negative；
- 学生成功：默认不对学生 token 做负更新；
- 学生成功而教师候选失败：交换 chosen/rejected，允许教师真正向学生的新发现学习；
- 二者都失败：跳过教师更新，除非有可信 process verifier。

这不会天然打乱师生角色。教师仍是主要正向参照，verifier 只在有外部证据时纠正少数角色误判。首轮必须同时跑无 verifier 与 verifier-gated 版本，才能知道 verifier 是必要条件还是锦上添花。

### 4.6 从 token 扩展到 Agent action

在工具 Agent 中，最自然的监督单位不是任意自然语言 token，而是一次完整 assistant turn 或 tool call：

sₜ = (user prompt，已有 assistant 消息，全部 tool observations，当前 workspace 摘要)

aₜ = 下一次回复或结构化 tool call

学生实际 action 作为 a⁻，教师在完全相同 sₜ 上产生的 action 作为 a⁺。这样仍然保持同状态，但偏好单位更接近真实决策。token 级数学实验只负责验证梯度机制；Agent action 级实验才直接检验长程动机。

## 5. 可证伪假设

- H1：标准 OPD 比只做 GRPO 更有效地改善学生在长响应上的局部决策。
- H2：无 verifier 的同状态 hard-negative 教师更新可以提高教师 outcome accuracy。
- H3：同状态 negative 显著优于 shuffled-state negative；否则“同状态”不是增益来源。
- H4：教师提升不应只表现为 entropy 下降，而应伴随旧错误重现率下降和 benchmark accuracy 上升。
- H5：动态教师下的学生不弱于冻结教师 OPD；否则飞轮没有形成，只是牺牲学生来锐化教师。
- H6：verifier gating 能减少错误伪标签，但不是无 verifier 方法产生收益的必要条件。

以下任一结果都应被如实视为反证或边界：

- 教师 margin 变大，但 MATH-500/AIME 不升反降；
- same-state 与 shuffled-state 没有可重复差异；
- 只有 verifier-gated 版本有效；
- 教师变强但学生变弱，或反之；
- 收益完全可由更低 entropy、更多计算或额外 teacher forward 解释。

## 6. 与现有工作的边界

截至 2026-08-27 的检索中，没有找到与“**强教师继续监督学生，同时强教师对学生同状态动作接受显式 hard-negative 参数更新**”完全相同的公开方法；这是检索结论，不是不存在证明。最接近的工作如下：

| 工作 | 做了什么 | 与本提议的关键差别 |
| --- | --- | --- |
| [GKD / 原始 OPD](https://arxiv.org/abs/2306.13649) | 在学生访问的状态上让学生拟合教师 | 教师冻结，只更新学生 |
| [G-OPD / ExOPD](https://arxiv.org/abs/2602.12125) | 广义化 OPD reward 与 reference，发布了可复现代码 | 仍以教师给学生提供分布信号为主；没有教师排斥学生 hard negative 的优化器 |
| [PLaD](https://aclanthology.org/2024.findings-acl.923/) | 把教师输出当 pseudo-positive、学生输出当 pseudo-negative 训练学生 | 负样本进入学生 preference loss；不是沿学生中间状态更新教师 |
| [CoPD](https://arxiv.org/abs/2604.27083) | 专家各自做 RLVR，并在训练中做双向 OPD、共同演化 | 最接近“双模型共同更新”，但核心是双向吸引与能力融合，不是教师对学生同状态动作做排斥更新 |
| [OPCoD](https://arxiv.org/abs/2606.14368) | 两个领域模型通过 peer feedback 与 self-distillation 互相提升 | 交换自然语言反馈并拟合带反馈 self-teacher，不是 source-asymmetric hard-negative teacher loss |
| [MOPD](https://arxiv.org/abs/2605.12652) | 用同一 prompt 下的成功、失败 peer rollouts 改善教师条件信号 | 失败轨迹作为教师上下文；没有让教师参数对学生 negative 反向传播 |
| [RLRT](https://arxiv.org/abs/2605.10781) | 学生成功且偏离教师时，反转教师信号以鼓励探索 | 更新对象仍是学生，不是外部教师 |
| [OPDVR](https://arxiv.org/abs/2608.24696) | 用可验证 reward 约束 OPD 信号符号并接入 RLVR | 对学生信号做 correctness gating，教师仍是固定参照 |
| [DUET](https://arxiv.org/abs/2608.14644) | 用正、负双教师的分歧给学生做 token 级偏好优化 | 推开的是学生相对负教师；没有训练正教师从学生 negative 学习 |

所以论文的新颖性不能写成“首次双向蒸馏”或“首次使用负样本”。更稳妥的主张是：

> 我们研究一种非对称、双优化器的 OPD：学生在自己的 on-policy 状态上吸引到教师，而可训练教师在这些完全相同的状态上排斥学生生成的 hard negatives，并由冻结初始教师约束漂移。

## 7. 初步实验规格

### 7.1 模型与算力

| 阶段 | 学生 | 教师 | 用途 | 建议资源 |
| --- | --- | --- | --- | --- |
| Phase 0：梯度 smoke | Qwen3-0.6B | Qwen3-1.7B | 只验证梯度方向、mask、checkpoint 与 worker 接线 | 2–4 张 GPU；不用于论文结论 |
| Phase 1：数学主实验 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500](https://huggingface.co/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500) | 最小强弱师生验证 | 8×80GB GPU 推荐；与 G-OPD 8 卡 recipe 对齐 |
| Phase 2：Agent | 先复用 1.7B → 4B；若出现能力地板，再升到 Qwen3-4B → 本地 Qwen3-8B | 同左 | 验证长程工具轨迹 | 只在 Phase 1 机制成立后投入 |

Phase 1 使用同一 Qwen3 tokenizer，关闭 thinking，避免额外 tokenizer 对齐与长思维模式混杂。G-OPD README 中教师写成了简写 `Qwen/Qwen3-4B-Non-Thinking-RL-Math`；实际公开权重使用上表的 `Keven16/...-Step500`。

原始 G-OPD 脚本设置 `trainer.n_gpus_per_node=8`、学生 1.7B、教师 4B、FSDP 与 teacher parameter offload。加入可训练教师和冻结 πT₀ 后显存需求更高，因此 8×40GB 不作为首轮承诺配置；若只有 40GB，先缩短 response、减小 micro-batch 并开启 optimizer/parameter offload。

### 7.2 训练数据

主数据使用 [Keven16/G-OPD-Training-Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data) 中：

`DeepMath-103K/train_filtered_level6.parquet`

数据仓库约 3.39 GB。实验分两步：

1. Pilot：固定 seed 42 打乱后取 10,000 个 train prompt，再取互斥的 1,000 个 dev prompt；只跑 1 epoch。
2. Confirmation：pilot 的机制对照通过后，再用完整 DeepMath-103K 文件跑 3 seeds（42、43、44）。

不要把 AIME 2024/2025 当训练中的 checkpoint-selection validation。G-OPD 原脚本每 10 step 测 AIME 适合复现原结果，但会导致我们自己的 test overfitting；本实验只用 DeepMath dev 选超参数，最终一次性解封 AIME/HMMT。

### 7.3 数学 benchmark 与指标

| Benchmark | 规模 | 作用 | 主指标 |
| --- | ---: | --- | --- |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500 | 样本量较大的主评测 | math-verify pass@1；按 level/subject 分层结果 |
| G-OPD AIME 2024 | 30 | 高难竞赛数学 | avg@32、pass@32、平均输出长度 |
| G-OPD AIME 2025 | 30 | 高难 OOD | avg@32、pass@32、平均输出长度 |
| G-OPD HMMT 2025 Feb | 30 | 高难 OOD | avg@32、pass@32 |
| G-OPD HMMT 2025 Nov | 30 | 高难 OOD | avg@32、pass@32 |

G-OPD 的 `math_eval/eval_math.py` 把 32 次生成中逐样本平均正确率打印为 `Accuracy`，本文称 **avg@32**；它把“至少一次正确”的比例打印为拼写有误的 `passs@k`，本文称 **pass@32**。二者不可混写。AIME/HMMT 每套只有 30 题，必须报告 3 seeds 和 paired bootstrap 95% CI，不能只报一个最好 seed。

除 outcome 指标外，必须同时记录机制指标：

- 教师与学生各自的 pass@1 / avg@32，而不只报学生；
- 教师相对初始 πT₀ 的每 token KL、entropy、top-k overlap；
- usable hard-negative coverage：有效 negative token 数 ÷ 有效 response token 数；
- 不同轨迹深度上的 teacher–student margin；
- 训练后在相同 state replay 时，教师重复原 student negative 的概率变化；
- pair outcome 四象限：T 对/S 错、S 对/T 错、都对、都错；
- 生成 token 数、GPU-hours、峰值显存与 wall time。

### 7.4 实验矩阵

所有方法匹配 prompt、student rollout 数、最大响应长度与随机种子。涉及教师更新的方法还要匹配 teacher optimizer steps；计算量更大的方法同时报告等 token 和等 wall-clock 两种比较。

| ID | 学生更新 | 教师更新 | 目的 |
| --- | --- | --- | --- |
| E0 | 无 | 无 | 初始 S₀/T₀ 基线 |
| E1 | GRPO / RLVR | 无 | 稀疏 outcome baseline |
| E2 | 标准 OPD | 冻结 | 验证沿轨迹正向局部监督 |
| E3 | OPD + RLVR | 冻结 | 排除“只是把两种已有信号相加” |
| E4 | OPD + RLVR | 同状态 student hard negative；无 verifier | 核心提议 |
| E5 | 同 E4 | shuffled-state student negative | 检验 same-state 必要性 |
| E6 | 同 E4 | 随机 negative / self-sharpening | 检验是否只是熵收缩 |
| E7 | 同 E4 | verifier-gated hard negative | 检验 verifier 的边际价值 |
| E8 | 冻结学生 | 同 E4 | 单独判断教师能否从学生 negative 获益 |

首轮 pilot 不必把所有超参数做网格搜索。先固定 β = 0.1、λHN = 0.1、k = 64；只有 E4 在 dev 上表现出机制迹象后，再对 λHN ∈ {0.05, 0.1, 0.2} 做一次小消融。

### 7.5 成功判据

只有同时满足以下条件，才认为核心假设得到支持：

1. E4 的教师在 MATH-500 与四套竞赛题聚合分数上优于 T₀，且 paired CI 不只是 entropy 收缩；
2. E4 稳定优于 E5/E6，说明同状态 student negative 提供了信息；
3. E4 的学生不弱于冻结教师的 E3；
4. 至少 3 seeds 中方向一致，而不是挑最好 checkpoint；
5. 旧 student negative 的复现率下降，并与 outcome error correction 正相关。

如果只有 E7 有效，则结论应改成“verifier 是必要的方向约束”，不能继续声称无 verifier 即可形成可靠飞轮。

## 8. Agent 长程验证

数学任务先验证优化机制；长程主张使用本仓库现有 harness 做第二阶段因果实验，不必一开始再引入一个大型 Agent 框架。

现有 `agent/tasks/synthesis_80k.jsonl` 包含 63 个模板：21 个 base task × short/medium/long 三种轨迹长度，元数据合计 80,000 rollouts。按 base task 分组切分，不能把同一任务的 short 放训练、long 放测试而伪装 OOD：

- 14 个 base task 训练；
- 3 个 base task dev；
- 4 个 base task test；
- seed 42 固定分组；测试重点单独报告 long profile。

`agent/tasks/harness_smoke.jsonl` 的 4 题只检查链路；`synthesis_full.jsonl` 的 21 题只作 harness 校准，不作为论文级公开 benchmark。这里的目的，是判断 same-state action negative 是否真的改善长程 credit assignment，而不是声称通用 coding-agent SOTA。

Agent 指标：

- verifier `task_success` / success@1 与 pass@8；
- protocol validity、tool-call parse rate、tool execution success rate；
- 失败动作后的 recovery success；
- 平均 turns、tool calls、generated tokens 与 task wall time；
- 按 short/medium/long 分层的成功率；
- hard-negative coverage 与 teacher margin 随 action depth 的曲线；
- harness fault、verifier fault 单独计数并从模型 reward 中剔除。

如果本地机制成立，再考虑公开 benchmark。SWE-bench Verified 成本高且 1.7B/4B 容易出现能力地板，不应作为第一轮环境搭建目标。

## 9. 复现环境

### 9.1 当前环境不能直接使用

截至 2026-08-27，现有环境：

`/home/JJ_Group/lih2511/.conda/envs/nanoagent`

的 `pip check` 不是 clean：

- 本地 `verl 0.10.0.dev0` 要求 transformers 5.5.3–5.10.x，但环境是 4.51.3；
- vLLM 0.8.5.post1 要求 OpenTelemetry 1.26.x，但环境是 1.44.0。

根目录旧 requirements、当前 `third_party/verl` 和 G-OPD v0.6.1 属于三套不同依赖世界。不要在 `nanoagent` 环境里继续覆盖包；新建独立环境。

### 9.2 需要 clone 的框架

首轮只需要 clone 一个外部框架；G-OPD 已内置其修改版 verl、数学 evaluator 和数据格式，不要再 clone TRL、OpenRLHF 或另一份 verl。

```bash
export OPD_PROJECT=/home/JJ_Group/lih2511/test/nanoGPT
export GOPD_ROOT=/home/JJ_Group/lih2511/test/nanoGPT/third_party/G-OPD

git clone https://github.com/RUCBM/G-OPD.git "$GOPD_ROOT"
git -C "$GOPD_ROOT" checkout 37371a4c31ad7947746200d234161769191f4748
git -C "$GOPD_ROOT" rev-parse HEAD
```

固定提交：`37371a4c31ad7947746200d234161769191f4748`。本仓库已有的 `third_party/verl` 可作为新版 OPD/teacher-loop 参考，但不是 Phase 1 的运行环境。

### 9.3 Conda 与包

G-OPD 官方 recipe 是 Python 3.10、verl 0.6.1、PyTorch 2.6/CUDA 12.4、vLLM 0.8.5.post1。为了减少依赖，首轮关闭未使用的 SGLang 和 Megatron：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda create --name tt-hn-opd python=3.10 -y
conda activate tt-hn-opd

cd /home/JJ_Group/lih2511/test/nanoGPT/third_party/G-OPD/verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh

python -m pip install \
  "transformers==4.51.3" \
  "math-verify==0.9.0" \
  "huggingface-hub[cli]==0.30.2" \
  "opentelemetry-api==1.26.0" \
  "opentelemetry-sdk==1.26.0" \
  "opentelemetry-exporter-otlp==1.26.0"
```

安装脚本还会安装这些运行时包：torch 2.6.0、torchvision 0.21.0、torchaudio 2.6.0、tensordict 0.6.2、FlashAttention 2.7.4.post1、FlashInfer 0.2.2.post1、Ray、datasets、accelerate、peft、pyarrow、pandas、hydra-core、wandb、liger-kernel、fastapi、pydantic 与 grpcio。

G-OPD 对 Ray 等少数包没有给出精确 pin。第一次 clean smoke 成功后，必须保存真实解析结果；在保存前不能声称 bitwise environment reproducibility：

```bash
mkdir -p /home/JJ_Group/lih2511/test/nanoGPT/artifacts/tt-hn-opd/env
python -m pip freeze > /home/JJ_Group/lih2511/test/nanoGPT/artifacts/tt-hn-opd/env/pip-freeze.txt
python -m pip check
```

运行入口必须位于 `third_party/G-OPD/verl`，让 Python 导入该目录内的 verl 0.6.1。不要在这个环境安装本仓库的 `third_party/verl`。

### 9.4 GPU preflight

当前登录环境没有可见 GPU，正式运行前在计算节点执行：

```bash
nvidia-smi
python -c "import torch, transformers, vllm, ray; print(torch.__version__, transformers.__version__, vllm.__version__, ray.__version__); print(torch.cuda.device_count()); assert torch.cuda.device_count() >= 8"
python -m pip check
```

只有 CUDA 可见、8 卡数量正确且 `pip check` 为 clean，才开始下载后的模型加载 smoke。仓库文档同时出现 air-node-02 与 air-node-03，不在脚本里硬编码节点名；提交作业前确认实际可用节点。

## 10. 权重与数据下载

以下 revision 均固定于 2026-08-27：

```bash
export OPD_MODEL_DIR=/home/JJ_Group/lih2511/test/nanoGPT/model/language_model/checkpoints/opd
export OPD_DATA_DIR=/home/JJ_Group/lih2511/test/nanoGPT/assets/data/opd

huggingface-cli download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$OPD_MODEL_DIR/Qwen3-0.6B"

huggingface-cli download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir "$OPD_MODEL_DIR/Qwen3-1.7B"

huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
  --revision 05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa \
  --local-dir "$OPD_MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step500"

huggingface-cli download Keven16/G-OPD-Training-Data \
  --repo-type dataset \
  --revision c9bc6783733858dd4892f26f95e4aea0942e91f1 \
  --local-dir "$OPD_DATA_DIR/G-OPD-Training-Data"

huggingface-cli download HuggingFaceH4/MATH-500 \
  --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --local-dir "$OPD_DATA_DIR/MATH-500"
```

0.6B 只供 Phase 0；若直接从 Phase 1 开始可以不下载。所有下载目录应保持在 `.gitignore` 内，只提交 revision、文件 hash、数据切分 manifest 与实验配置，不提交权重本体。

## 11. G-OPD 基线复现

先不改算法，确认原始 OPD 能跑通。官方入口是：

`third_party/G-OPD/verl/examples/g_opd/run_qwen3-4b-g-opd.sh`

至少修改以下路径，不要照抄 README 中不存在的教师简写：

- `data.train_files` → 本地 `train_filtered_level6.parquet`；
- `data.val_files` → pilot dev，不能用 AIME 选 checkpoint；
- `actor_rollout_ref.model.path` → 本地 Qwen3-1.7B；
- `actor_rollout_ref.ref.model.path` → 本地 Step500 教师；
- pilot 将 `data.max_response_length` 从 16384 降到 4096；
- `trainer.logger='["console"]'` 或 `WANDB_MODE=offline`，不要在脚本中保存 API key。

基线验收顺序：

1. 32 prompts、max response 512、1 个 optimizer step；
2. 256 prompts、max response 2048、保存并恢复 checkpoint；
3. 10K pilot、max response 4096；
4. 只有前三步通过后才运行完整 103K。

数学 evaluator 示例：

```bash
cd /home/JJ_Group/lih2511/test/nanoGPT/third_party/G-OPD/math_eval
CUDA_VISIBLE_DEVICES=0,1 python eval_math.py \
  --input_file ../data/aime24/test.jsonl \
  --model_path /path/to/checkpoint \
  --output_file ./eval_outputs/aime24/model.jsonl \
  --max_tokens 16384 \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_num_seqs 256 \
  --n 32 \
  --seed 42
```

对 AIME25、HMMT25 Feb、HMMT25 Nov 替换 input/output 路径重复运行。MATH-500 需一次性转换为 evaluator 所需的 `{"problem": ..., "answer": ...}` JSONL；转换脚本必须保存源 revision 和行数 500 的 assert。

## 12. 最小实现改动

不新建训练框架，也不先写通用 registry。直接在 G-OPD 已有 ref worker 上增加一个可选的 teacher optimizer。

需要触碰的核心位置：

- `verl/verl/workers/config/actor.py` 与 `verl/verl/trainer/config/actor/actor.yaml`：增加 `teacher_hn.enabled/lr/beta/weight/top_k/update_every`；
- `verl/verl/workers/fsdp_workers.py`：在开关打开时让 ref teacher 可训练，创建独立 optimizer，暴露 `compute_teacher_targets` 与 `update_teacher_hn`；
- `verl/verl/workers/actor/dp_actor.py`：实现两 token gather、mask、参考 margin 与 LHN；
- `verl/verl/trainer/ppo/ray_trainer.py`：固定“先 teacher score，后 student update，再 teacher update”的顺序并记录指标；
- `verl/examples/g_opd/`：新增一个实验脚本，不修改官方 baseline 脚本，保证 E2 可原样复现。

每个 batch 新增的张量只需要：

- `teacher_positive_ids`，形状 [B, L]；
- `student_negative_ids`，可直接复用 response ids；
- `teacher_hn_mask`，形状 [B, L]；
- `teacher_ref_margin`，形状 [B, L]。

不要跨模型传梯度，不保存完整 vocabulary logits，不为首轮引入 TRL/DPOTrainer。

### 12.1 最小可运行检查

非平凡梯度逻辑必须留下一个小测试，不引入测试框架：

1. 构造两 token toy logits；一步 LHN 后断言 log p(a⁺) − log p(a⁻) 增大；
2. 断言 masked position 权重不变；
3. 断言 πT₀ 与学生参数没有 teacher-loss gradient；
4. 32-sample integration smoke 中，教师 checkpoint 确实变化、πT₀ hash 不变、学生用的是更新前 teacher score；
5. checkpoint save/resume 后一小步 loss 与未中断 run 在容差内一致。

## 13. 执行顺序

- [ ] 新建独立环境并让 `pip check` clean。
- [ ] clone/pin G-OPD，下载固定 revision 的权重与数据。
- [ ] 跑初始 S₀/T₀ 的全部 benchmark，先确认教师确实整体强于学生。
- [ ] 原样复现 E2 标准 OPD。
- [ ] 实现 teacher hard-negative loss 与 toy gradient check。
- [ ] 跑 E4/E5/E6 的 10K pilot，只看 DeepMath dev。
- [ ] 冻结配置后评测 MATH-500、AIME24/25、HMMT25 Feb/Nov。
- [ ] 3 seeds 确认；失败也保留完整日志。
- [ ] 数学机制成立后，接入 action 级 Agent 实验。
- [ ] 最后再决定是否加入 verifier gating、扩大模型或上公开 Agent benchmark。

## 14. 检索方法与来源限制

本次检索以 arXiv 论文、作者/机构页面、官方 GitHub 与 Hugging Face 仓库为主，围绕以下组合查询：on-policy distillation、bidirectional/co-evolving distillation、student failure/negative sample、trainable teacher、verifiable reward、peer feedback 与 hard negative。重点对照了 CoPD、OPCoD、MOPD、RLRT、OPDVR、DUET、PLaD 和 G-OPD。

“没有找到完全相同方法”是截至 2026-08-27 的检索推断，受预印本更新、命名差异与未公开工作限制。投稿前必须重新检索，尤其检查 2026 年 8 月之后的 OPD/RLVR 论文。框架与权重的可复现事实以以下一手来源为准：

- [G-OPD 官方代码](https://github.com/RUCBM/G-OPD)
- [G-OPD 论文](https://arxiv.org/abs/2602.12125)
- [Qwen3-1.7B 权重](https://huggingface.co/Qwen/Qwen3-1.7B)
- [4B RL Math 教师权重](https://huggingface.co/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500)
- [G-OPD 训练数据](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data)
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
