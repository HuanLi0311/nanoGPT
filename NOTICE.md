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

关于将 Skill Internalization 扩展到 Harness Internalization 的合作想法

学长您好！

我最近阅读了您关于 SKILL0 和 SKILL1 的工作，很受启发。相关工作展示了一个重要方向：模型可以在训练早期借助外部 skill 完成任务，随后通过合适的训练过程逐步减少对 skill 的依赖，将其中一部分能力内化到模型自身。

在此基础上 我最近在思考模型能否逐渐内化的不只是外部 skill 提供的知识或程序性经验，也包括 Agent harness 所承担的部分工具调用与回溯职责？在当前的 Agent 系统中，harness 往往不仅负责工具调用和消息传输，还会参与规划、上下文管理、反思、错误恢复以及流程控制等工作。我们的初步想法是：在后训练早期，使用较厚的 harness 帮助模型稳定完成任务；随着模型能力提升，再逐步减少其中部分语义支撑，使模型接管原本由 harness 完成的工作。训练完成后，部署时只保留必要的工具、权限、sandbox 和验证等基础设施，而尽量减少对太重harness依赖 从而将该能力内化进模型侧。我觉得这可能是对 skill internalization 的一个自然延伸。

由于skill0/skill1这些先前的工作在类似方面已经积累了很好的经验 所以觉得给您发邮件会比较合适。若您有相关想法 我很希望能参与进咱们组当中 若您暂时没有相关想法 也希望能和您交流学习！

  感谢您们阅读这封邮件。期待您的反馈！

  祝好，
  [你的名字]
  [单位/实验室]
  [联系方式]