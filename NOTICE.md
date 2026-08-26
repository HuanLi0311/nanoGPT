# Harness 泛化之后：让模型逐渐不再依赖 Harness

> Proposal motivation draft

## 1. 从 Harness 过拟合开始

Agentic RL 中，一个经常被忽略的问题是：模型学到的究竟是“如何解决任务”，还是“如何在某一个固定 harness 的约定下解决任务”。如果训练轨迹始终由同一个脚手架生成，模型可能把工具格式、历史组织方式、反思节奏和停止条件当成任务本身的一部分。KAT-Coder-V2.5 在 Harness Scaling 部分将模型对 Agent 脚手架的依赖归纳为三类过拟合：一是**格式过拟合**，模型固化于特定工具调用格式，更换协议后解析失效；二是**上下文结构过拟合**，依赖训练时固定的历史排布，上下文经过压缩、截断重组后性能下降；三是**控制流过拟合**，依靠外部脚手架提供反思、终止条件，缺少外部规划就无法自主推进任务。该工作认为 Harness 扩容不能只堆砌脚手架数量，要围绕工具协议、上下文策略、控制流复杂度这三个有效维度做多样化，在 RL rollout 中混用白盒、黑盒 harness，把评估重心从基准分数转向判别模型学到的是任务本身，还是脚手架的接口习惯。

几份大模型 Agent 相关报告分别从不同角度揭示模型对外部脚手架（harness、协议、上下文、控制流）的依赖问题：

- **Qwen3-Coder-Next**：证明训练时使用多样化输出格式，能够缓解模型对单一工具调用模板的过拟合，提升格式鲁棒性，但没有探究模型能否内化规划、反思这类外部组件。
- **Kimi K3**：做到训练时动态切换、组合各类 Agent 模块，弱化任务层面的脚手架依赖，但底层消息传输协议依旧存在硬绑定；模块多样化不等于摆脱协议约束。
- **DeepSeek V3.2/V4**：指出上下文管理策略直接构成模型工具使用的能力边界，现有方案是靠框架设计上下文留存规则，模型本身并不能自主处理上下文被篡改、缺失的情况。
- **Kimi K2.5**：借助 Agent Swarm 将复杂任务的控制流交给外部编排层实现任务扩容，但也留下疑问：模型是真正掌握了任务分解与调度，还是单纯依赖外部多智能体框架？移除编排层后能力是否会衰减？

KAT-Coder-V2.5：[Harness Scaling](https://arxiv.org/abs/2607.05471)

## Motivation

KAT-Coder-V2.5 给出了一个重要的诊断：如果模型只在单一 Harness 中训练，它可能学到的不是任务本身，而是特定的格式、上下文结构和控制流协议。通过扩大 Harness 的分布，可以减轻这类过拟合，使模型适应更多不同的脚手架。但这种视角主要把 Harness 当作模型需要泛化到的外部接口，仍然没有回答一个更根本的问题：模型在一个更强的 Harness 下取得的能力，究竟有多少已经转移到了模型自身？

在实际的 Agent 系统中，Harness 承担的不只是工具调用和消息传输，还可能代替模型完成规划、上下文管理、反思、错误恢复和停止判断等语义工作。因此，当模型在复杂任务上表现变好时，性能提升可能来自模型能力，也可能只是来自更厚的外部控制。我们希望把这种“外部支撑是否被模型接管”单独拿出来研究，并将其视为与 Harness 格式、上下文结构和控制流多样性不同的另一个问题维度：Harness Scaling 改变模型面对的 Harness 分布，而我们的方向改变训练过程中 Harness 对模型提供的语义支撑强度。

我们的基本设想是让 Harness 在后训练过程中逐渐由厚变薄。训练早期，模型的推理和错误恢复能力尚不充分，可以由较厚的 Harness 提供稳定的上下文、反馈、反思和流程控制；随着这些行为逐渐被模型学会，再有选择地撤除相应的语义组件，迫使模型在缺少外部代劳的情况下接管原有职责。训练完成后，部署时只保留工具调用、权限控制、sandbox、存储和结果验证等必要的基础设施，而不再依赖完整的语义控制层。这里的目标不是让模型适应某一种更薄的固定模板，而是检验原本由 Harness 提供的能力是否真正内化为模型策略。

因此，本文关注的核心问题不是“模型能否适配更多 Harness”，而是“模型能否在训练过程中逐渐需要更少的 Harness 帮助”。这个问题与 Harness Scaling、skill internalization 以及自动化 Harness editing 都有联系，但并不等同。下面将分别梳理这些方向：它们如何扩大 Harness 分布、如何把外部 skill 的作用转移给模型，以及如何对 Harness 进行适配、修复或自动演化，并据此明确本文的研究位置。

## 3. 相关工作：从 Harness 多样化到外部能力内化

上面的工作主要研究两件事：扩大模型接触到的任务、环境和 Harness 分布，或者让固定模型在更强的运行时脚手架下完成任务。与本文最接近的工作，是另一条正在形成的 **skill internalization** 研究线：它们把外部 skill 看作训练早期的程序性知识支架，再尝试让模型在没有 skill 注入时继续完成任务。

### 3.1 Skill internalization：最接近我们的先验

[SKILL0](https://arxiv.org/abs/2604.02268) 是目前与本文最接近的工作。它在训练 rollout 中提供完整的外部 skill context，然后通过 Dynamic Curriculum 逐步减少 skill budget；模型定期在有 skill 和无 skill 的条件下比较表现，根据每个 skill 对当前策略的 helpfulness 进行筛选，最终转向完全 zero-shot 的执行。SKILL0 明确把“从依赖外部 skill 到自主完成任务”作为训练目标，并提供了[公开代码](https://github.com/ZJU-REAL/SkillZero)。这说明“先用外部支架稳定探索，再撤掉支架促使模型内化”本身是一个有直接先例的训练范式，而不是单纯的直觉。

[Skill0.5](https://arxiv.org/abs/2605.28424) 进一步指出，并非所有 skill 都应该被同样处理：通用 skill 可以被内化，任务特定 skill 则可以继续作为运行时资源使用，并通过难度感知的路由器在两者之间选择。它强调的是内化和利用之间的折中，以及对分布外任务的帮助；本文则关心 Harness 中由外部系统承担的语义职责能否逐步迁移到模型，而不只区分通用知识和任务知识。

[SKILLC](https://arxiv.org/abs/2605.27899) 把有 skill 与无 skill 的 rollout 配成对照，并把两者的差异直接用于 contrastive credit assignment，试图区分“依赖 skill 才成功”和“已经能够自主成功”的策略更新。[OPID](https://arxiv.org/abs/2606.26790) 和 [SEED](https://arxiv.org/abs/2607.14777) 则从 on-policy trajectory 中提取 hindsight skill，再通过蒸馏或辅助目标把其行为效果沉淀回模型。这些工作主要改进的是**能力内化的学习信号**；它们没有把 planner、reflection、summary、retry 或 subagent orchestration 等更广义的 Harness 控制组件作为可逐步撤除的对象。

[Skill1](https://arxiv.org/abs/2605.06130) 和 [SkillRise](https://arxiv.org/abs/2607.26784) 研究 skill 的选择、使用、提炼和跨任务复用，使外部 skill library 持续演化。它们关注的是如何更好地管理和利用外部技能，而不是在训练后让模型摆脱整个技能层。因此，本文与 skill internalization 工作存在明确继承关系，但研究范围从“外部知识提示是否被吸收”扩展到了“外部语义控制是否被接管”。

### 3.2 Runtime Harness 的适配、修复与自动演化

[Life-Harness](https://arxiv.org/abs/2605.22166) 在冻结模型参数的前提下，从失败轨迹中提取环境契约、程序性 skill、动作实现和轨迹调节方面的 runtime intervention，并把改进后的 Harness 固定下来用于后续任务。它证明了不改模型也可以通过适配接口提升 Agent 表现，但其目标是**让运行时 Harness 替模型补偿能力**，而本文要研究的是训练过程中逐步减少这种补偿。

[HarnessFix](https://arxiv.org/abs/2606.06324) 把失败轨迹和 Harness 实现对齐，构造 Harness-aware Trace Intermediate Representation，再将诊断结果映射到受限的 repair operator，并用回归验证接受或拒绝 patch。它为安全地修改 Harness 提供了很有价值的工程范式：故障定位、局部编辑、回归检查和失败回滚。但 HarnessFix 的目标是修复有缺陷的 Harness，而不是有意移除一个原本有效的语义组件。

[Harness-R1](https://arxiv.org/abs/2608.02276) 更直接地训练一个专门的 Harness engineer：它读取目标 Agent 的失败轨迹，生成可执行 patch，再通过重新运行冻结的目标 Agent 获得真实任务收益作为奖励。它证明了 executable Harness editing 可以被作为一个学习问题处理，并且与目标模型形成协同演化。但该方向目前仍是很新的工作；其目标是提升冻结 Agent 的成功率，公开 patch 接口也主要围绕生命周期 hook 的新增或覆盖，而不是构造“由厚变薄”的单向编辑过程。

[Agentic Harness Engineering](https://arxiv.org/abs/2604.25850)、[Meta-Harness](https://arxiv.org/abs/2603.28052)、[HarnessX](https://arxiv.org/abs/2606.14249) 和 [MemoHarness](https://arxiv.org/abs/2607.14159) 则把 Harness 作为可搜索、可组合、可适配的优化对象：它们尝试从轨迹、分数和经验库中修改 prompt、tool、memory、middleware、workflow 或其他控制维度。这些工作说明 Harness 可以被显式表示并自动演化，但它们优化的是“如何得到更强或更适配的 Harness”，不是“如何让模型在训练中逐渐不再依赖 Harness”。如果直接采用其中的完整自动演化循环，Harness 编辑器本身就会成为另一项研究贡献，容易掩盖本文真正要回答的问题。

### 3.3 与本文的区别和研究定位

本文与上述工作的关系可以概括为：

- **与 KAT-Coder 的区别**：KAT-Coder 通过改变 Harness 的格式、上下文组织和控制流，增加模型面对的 Harness 分布，目标是跨 Harness 泛化；本文改变的是外部语义支撑的强度，目标是能力从 Harness 向模型迁移。前者回答“模型能否适应更多 Harness”，后者回答“模型能否在更少 Harness 帮助下完成同一类任务”。
- **与 SKILL0 的关系**：SKILL0 是本文最直接的先验，已经证明了“训练期注入、逐步撤除、最终 zero-shot 执行”这一基本范式。本文不能再把这个范式本身作为唯一贡献；需要把研究对象明确为 skill 之外的 Harness 语义组件，例如 Harness 强制触发的 reflection、自动 planner、summary、retry 或 subagent 编排，并研究它们的职责是否被模型接管。
- **与 Life-Harness、HarnessFix 和 Harness-R1 的区别**：这些工作主要让 Harness 适应模型、修复 Harness 缺陷或通过编辑增加任务收益；本文要求 Harness 的变化具有明确的“变薄”方向，不能通过新增补丁抵消被移除的支架，并且最终部署时保留的是轻量的基础设施，而不是完整的语义控制层。
- **与 Skill0.5、SKILLC、OPID 和 SEED 的区别**：这些工作解决的是哪些 skill 应该保留，以及如何给 skill internalization 提供更好的信用分配或蒸馏信号；本文关注的核心问题是 Harness 组件承担的规划、反思、状态管理和错误恢复职责能否内化。它们可以作为训练信号或课程调度的参考，但不应成为本文同时要解决的第二个算法问题。

因此，本文的最小研究边界应当是：**固定任务环境、工具执行和底层协议，只选择一个语义 Harness 组件族进行训练期 fading，观察模型是否能在该组件被撤除后接管其原有职责。** 如果只研究 skill context 的撤除，问题会与 SKILL0 高度重合；若希望保留独立性，更适合从 control-flow 或 context-management 组件开始，而不是同时改变 Harness 的格式、环境和控制流。

因此，我们要研究的核心问题是：

> **能否把 harness 作为训练早期的外部教师和稳定器，再在训练过程中逐渐撤掉其语义支撑，使模型在最终的轻量 harness 下自主完成原本需要厚 harness 才能完成的长程任务？**

在这个视角下，harness、模型和任务环境不是彼此替代的关系，而是一个逐步迁移能力归属的协同系统：强 harness 帮助较弱模型进入复杂任务，训练把其中一部分能力沉淀到模型，变薄后的 harness 再把模型暴露给新的任务难度和新的环境变化。


## harness改动方法

> 暂不展开具体的 Harness 编辑器、代码生成或自动改写方法；当前先固定研究问题、baseline 和评测协议。

## Baseline

这里的 baseline 不应主要比较不同的基础模型，而应在**相同基础模型、相同任务划分、相同工具和环境、相同优化预算**下，只改变目标语义组件在训练过程中的可用性。这样才能判断收益来自 Harness 的渐进撤除，还是来自模型规模、任务暴露量或额外计算量。

设目标组件为某一类语义支撑（例如自动 reflection、planner、summary、retry 或 context management），最小 baseline 集合如下：

1. **Always-thick**：训练和测试始终提供完整 Harness。它提供外部支撑下的性能上限，但不能说明能力已经被模型内化。
2. **Always-thin**：从训练开始就不提供目标组件，训练和测试都使用轻量 Harness。它检验模型能否在没有外部支撑的情况下从头学会任务，也作为“直接自主训练”的对照。
3. **Post-hoc removal**：训练阶段始终使用完整 Harness，只在测试时移除目标组件。它用于区分普通的 Harness-assisted post-training 与有意设计的 fading；如果这个 baseline 在轻量 Harness 下已经表现良好，渐进撤除的必要性就会减弱。
4. **Random/mixed availability**：从训练一开始就随机决定每条 rollout 是否提供目标组件，使其平均支持预算与 fading 方法相当。它控制“接触过薄 Harness”这一因素，用来检验由厚到薄的训练顺序是否重要。
5. **Harness-diversity control**：保持语义支撑始终完整，只改变格式、上下文组织或控制流协议，作为 KAT-Coder 风格的 Harness 多样性对照。它可以区分“模型适应了更多 Harness”与“模型需要更少 Harness 帮助”这两个不同来源的收益。
6. **Progressive fading（本文方法）**：训练早期使用较厚 Harness，随后逐步降低目标组件的支持强度，并在最终轻量 Harness 下测试。具体如何编辑或自动生成 Harness 暂放在后续部分。

如果最终选择的研究对象是 skill context，应额外复现 SKILL0 的 curriculum 作为最近邻 baseline；如果研究对象是 planner、reflection、summary、retry 或 context management，则不必把 SKILL0 变成第二个完整算法，只需复用其“厚支撑到零支撑”的对照思想。

所有 baseline 至少需要控制基础模型、训练样本或环境实例数、RL 更新步数、随机种子和最大 rollout 预算；Harness 增加的输入 token、模型生成 token、工具调用次数和 wall-clock 成本应单独记录，不能把更高的训练或推理成本误报为能力提升。

## 测评方案与benchmark

### 评测核心：同一任务上的 Harness 支撑曲线

第一版不新造一个独立任务集，而是在已有可执行环境上为同一批任务提供多种 Harness 条件。固定任务初始状态、用户目标、工具集合、权限和底层环境，只改变目标语义组件的支持程度：

- **Full Harness**：目标组件完整工作，例如自动提供规划、反思、摘要、重试或上下文管理；
- **Partial Harness**：只保留必要的触发信号或基础反馈，减少 Harness 对决策的代劳；
- **Light Harness**：只保留工具执行、权限、sandbox、存储和结果验证等基础设施，目标语义组件不再主动替模型做决定。

在训练的多个 checkpoint 上，使用完全相同的 held-out 任务分别运行这三种条件，形成 (P(	ext{success}mid	ext{checkpoint},	ext{Harness level})) 曲线。主要结果不是单一的最终分数，而是：

- **Light-Harness success**：最终轻量 Harness 下的任务成功率，是部署目标对应的主指标；
- **Support gap**：Full Harness 与 Light Harness 的成功率差距。随着训练进行，若差距缩小且 Light-Harness success 上升，才说明模型可能接管了外部组件的职责；
- **相对 baseline 的内化增益**：重点比较 Progressive fading 与 Post-hoc removal、Random/mixed availability 的 Light-Harness 表现，而不是只与 Always-thin 比较。

为了避免把“学会某种提示模板”误判成能力内化，还需要在不改变任务语义的情况下保留一组未见过的 Harness 实现，例如不同的提示措辞、上下文排列或等价的控制流实现。真正的内化应当在这些 implementation-held-out 条件下仍然成立；如果只对训练中出现过的薄模板有效，更可能是协议适应而不是 Harness 职责迁移。

### 指标

评测应优先使用环境执行结果，而不是只用语言模型 judge：

- **任务正确性**：最终数据库、文件系统、应用状态或代码测试是否满足目标；
- **副作用与约束遵守**：是否产生未授权修改、错误工具调用、违反 policy 的动作或不应出现的状态变化；
- **长程执行质量**：成功前的模型 turn 数、工具调用数、生成 token、失败恢复次数和最终成功率；
- **恢复能力**：在工具报错、返回信息不完整、上下文被压缩或中途出现干扰后，模型能否继续完成任务；
- **可靠性**：对同一任务重复运行多次，报告均值、方差和 pass^k，而不是只报告一次成功率。

其中，轨迹中的“模型是否输出了类似 reflection 的文字”只能作为辅助分析，不能作为主要指标。模型可能用不同形式完成同一语义职责，最终状态、关键中间里程碑和不应发生的事件更适合作为判断依据。

### 主 benchmark：AppWorld

主 benchmark 建议采用 [AppWorld](https://arxiv.org/abs/2407.18901)。它提供可控的多应用环境，包含 9 个日常应用、457 个 API 和 750 个交互式任务；官方评测使用基于环境状态的 unit tests，同时检查 collateral damage，因此允许不同的有效执行路径，不要求模型复现某一条固定 API 序列。官方仓库也提供 `train`、`dev`、`test_normal` 和 `test_challenge` 划分。[代码仓库](https://github.com/StonyBrookNLP/appworld)

AppWorld 适合当前问题的原因是：任务需要跨应用、多步 API 交互和持续状态管理，足以暴露 planner、context management、reflection 和 retry 等语义 Harness 的作用。训练或 curriculum 只使用 train/dev，最终报告在未参与训练的 test_normal 和 test_challenge 上进行；同一测试任务在 Full、Partial、Light 三种 Harness 下成对运行。

### 外部验证：ToolSandbox

为避免结论只适用于 AppWorld 的 API 形式，建议使用 [ToolSandbox](https://arxiv.org/abs/2408.04682) 做外部验证。ToolSandbox 包含有状态工具、工具之间的隐式状态依赖、多轮用户模拟，以及基于 Milestone/Minefield 的中间和最终执行评测；其场景平均交互轮数和工具调用数也高于许多单轮 tool-use benchmark。[代码仓库](https://github.com/apple/ToolSandbox)

ToolSandbox 特别适合检查模型是否能够在缺少自动反思或自动恢复时跟踪状态、处理工具异常、回溯错误并避免幻觉式调用。这里不需要把 ToolSandbox 作为第二个训练环境，第一版只将其作为 held-out evaluation，检验在不同任务分布和不同评测器上的 Light-Harness 泛化。

### 可选的可靠性检查：τ-bench

如果实验资源允许，可在 [τ-bench](https://arxiv.org/abs/2406.12045) 上补充低成本的可靠性检查。τ-bench 通过领域 API、策略约束和用户模拟测试 Agent，并用对话结束时的数据库目标状态判断任务是否完成，同时提供 pass^k 衡量多次运行的一致性。它适合检查 Harness 变薄后模型是否出现更多违规动作、错误确认或随机失败，但不作为本文的主 benchmark。

### 最小可行实验组合

第一版可以收敛为：**一个基础模型、一个目标 Harness 语义组件、AppWorld 主 benchmark、五个对照条件加一个 fading 方法、Full/Partial/Light 三档测试条件，以及 ToolSandbox 的 held-out 外部验证。** 这样评测的核心是“随着训练进行，模型在轻量 Harness 下是否获得更高且更稳定的执行能力”，而不是同时研究新的 Harness 编辑器、新的任务环境和新的 benchmark 构造方法。
