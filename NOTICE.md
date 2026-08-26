# Harness 泛化之后：让模型逐渐不再依赖 Harness

> Proposal motivation draft。本文档当前只讨论问题背景、相关工作和研究动机；评测设计与 harness 改写方案留到后续。

## 1. 从 Harness 过拟合开始

Agentic RL 中，一个经常被忽略的问题是：模型学到的究竟是“如何解决任务”，还是“如何在某一个固定 harness 的约定下解决任务”。如果训练轨迹始终由同一个脚手架生成，模型可能把工具格式、历史组织方式、反思节奏和停止条件当成任务本身的一部分。

KAT-Coder-V2.5 Technical Report（[arXiv:2607.05471](assets/paper/2607.05471.pdf)）在 §4.1 Harness Scaling 中把这种依赖拆成了三个方面：

1. **Format overfitting**：模型绑定某种工具调用格式；一旦从结构化 function calling 换成文本代码块、tag-based protocol 或其他协议，解析失败率显著上升。
2. **Context-structure overfitting**：模型依赖训练 harness 特定的历史拼接顺序；当上下文被滑动窗口、摘要压缩或不同的 observation truncation 策略重新组织时，行为发生退化。
3. **Control-flow overfitting**：模型依赖 harness 提供的反思时机和停止条件；当外部脚手架不再显式规划、反思或决定何时终止时，模型无法自主推进任务。

KAT-Coder 的关键判断是，Harness Scaling 的重点不在于简单增加 harness 数量，而在于让多样性落在真正有利于泛化的维度上。它因此把工具调用协议、上下文管理策略和控制流复杂度作为三条变化轴，并在 RL rollout 中引入不同的白盒和黑盒 harness。

这一步把问题从“模型在某个 benchmark 上得了多少分”推进到了更具体的问题：模型是否学会了任务本身，而不是某个脚手架的接口习惯。

## 2. 其他工作的补充视角

### 2.1 Qwen：格式多样性可以缓解协议依赖

Qwen3-Coder-Next 报告（[2603.00729](assets/paper/2603.00729.pdf)）主要聚焦 Format 这一维。报告指出，许多模型使用单一的 tool chat template 训练，因而容易过拟合到某一种输出结构；部署到未见过的 tool-calling schema 后，鲁棒性会下降。

Qwen 通过引入自然语言工具描述、JSON、Python 风格调用、XML 风格 schema 和 TypeScript 风格接口等多种格式，试图让模型学习与具体表面语法无关的工具使用行为。它还报告了 template 数量的受控变化：在数据量和训练配置保持不变的情况下，训练格式的多样性增加，SWE-bench Verified 上的格式鲁棒性随之提升。

这说明 Format overfitting 并不是一个纯粹的部署问题，训练分布本身可以改变模型对协议的依赖。但 Qwen 的重点仍然是“适应更多格式”，并没有进一步追问：如果不再提供某些外部的规划、反思或技能组件，模型能否接管这些组件的功能。

### 2.2 Kimi K3：可以随机化模块，但协议边界仍然可能很硬

Kimi K3 Technical Report（[Kimi-K3.pdf](assets/paper/Kimi-K3.pdf)）在 §4.2.1 中把 agent harness 表述为可配置、可组合的模块集合，包括工具接口、system prompt、context management、skills、memories 和 subagents。通过动态组合这些模块，训练可以覆盖 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 以及新的 harness。

这比只改变工具格式更进一步：它把 harness 当成训练分布的一部分，试图减少模型对单一脚手架约定的依赖。

但 K3 的 Appendix F 又显示出另一面。K3 在 thinking 模式下要求 preserved thinking history：`think` channel 需要保留在历史中，tool call 与 tool result 也需要按照固定的消息结构和索引关系回传。也就是说，任务层面的 harness 模块可以变化，模型面对的 wire protocol 却仍然可能存在硬耦合。

这带来一个重要区分：任务环境和 harness 的多样化，可以促使模型学习更一般的行为；但它并不自动意味着模型已经摆脱了外部协议、上下文管理或控制流的依赖。

### 2.3 DeepSeek：上下文管理本身会成为能力边界

DeepSeek-V3.2 报告（[DeepSeek-V3.2.pdf](assets/paper/DeepSeek-V3.2.pdf)）专门讨论了 tool-use 场景中的 thinking context management。它保留工具交互之间的历史 reasoning，但在新的 user message 到来时丢弃 reasoning；同时保留工具调用及其结果。报告还指出，某些把工具交互模拟成 user message 的 agent framework 可能无法触发这一上下文路径，因此不能充分受益于该机制。

DeepSeek-V4（[DeepSeek-V4.pdf](assets/paper/DeepSeek-V4.pdf)）进一步改变了 thinking history 的保留策略，并使用固定的工具调用 schema。这些工作说明，模型是否能持续推进长程任务，不仅取决于任务难度，也取决于 harness 如何组织和回传上下文。

不过，这类工作主要是在设计更有效的上下文策略，而不是研究模型能否最终内化上下文管理能力。一个能够在 preserved history 下工作的模型，未必能在历史被压缩、重排或部分缺失时自行恢复状态。

### 2.4 Kimi K2.5：控制流可以被外部化为 Agent Swarm

Kimi K2.5（[Kimi-K2.5.pdf](assets/paper/Kimi-K2.5.pdf)）通过 Agent Swarm 和 PARL，把任务分解、subagent 创建、并行调度和结果汇总外部化为可学习的控制流。模型需要决定是否拆分任务、如何委派以及何时并行执行。

这类方法说明，强 harness 可以把单个模型难以承担的复杂控制流拆成多个外部角色，从而扩大可处理任务的范围。但它也提出了新的问题：模型学到的是自主分解和调度，还是学会了依赖外部 subagent 来完成分解和调度？如果拿掉并行编排层，原有能力还能保留多少？

### 2.5 GLM-5：环境规模化不等于接口随机化

GLM-5（[GLM-5.pdf](assets/paper/GLM-5.pdf)）展示了大规模可验证环境、软件工程任务和异步 rollout 基础设施的扩展。它是一个很有用的对照：环境数量、仓库规模和任务复杂度的增加，能够扩大模型接触到的世界状态和长程任务分布，但并不自动改变模型所面对的 harness 协议、上下文结构或控制流。

因此，“环境规模化”和“接口随机化”是两个正交方向。一个模型可以在数万种环境中训练，却仍然只通过一种固定的工具 schema、历史模板和外部控制流与环境交互。

## 3. 现有问题留下的空缺

把这些工作放在一起，可以看到目前至少有两种常见做法：

- 增加任务和环境的复杂度，让模型在同一个 harness 下形成更长的推理和执行轨迹；
- 增加 harness 的格式、上下文和控制流多样性，让模型适配更多脚手架。

但这两种做法都没有直接回答另一个问题：**模型是否正在把 harness 原本承担的能力迁移到自身？**

如果一个模型在更复杂的环境中产生了更长的 trajectory，但 planner、summary、reflection 和 retry 仍然由 harness 自动提供，那么变长的可能只是外部系统，而不是模型自身的长程能力。

如果一个模型训练过很多种工具协议，但每一种 harness 都仍然提供完整的规划和反思，那么它可能获得了格式不变性，却仍然依赖厚重的控制脚手架。

反过来，如果从一开始就把所有判断都交给模型，训练又很容易受到幻觉、错误恢复失败和稀疏奖励的影响。把状态、规则、校验和流程全部程序化，虽然可以提高早期训练的稳定性，却可能把模型锁定在某个固定的操作范式中。

这似乎形成了一个钟摆：

```text
完全程序化  ←────────────────→  完全自主
稳定、可验证                         灵活、可泛化
但容易僵化                           但容易漂移和发散
```

真正值得研究的可能不是在两个端点之间选择一个静态折中点，而是让这个折中点随着模型能力的发展而移动：早期需要更多外部支撑，后期逐步把支撑撤掉。

## 4. 我们的动机：训练期的 Harness Fading

我们希望研究一种训练期的 harness 演化过程：在模型推理能力尚不充分时，使用较厚的 harness 稳定任务执行、提供必要的上下文和反馈；随着模型逐渐学会规划、反思、状态管理和错误恢复，再逐步移除一部分语义脚手架，迫使模型接管这些原本由 harness 完成的工作。

这里的“变薄”不是把所有外部系统都删掉，也不是简单减少代码量，而是减少 harness 对决策的代劳。例如：

- 从自动调用 skill，过渡到模型自主判断是否需要该 skill，最后在没有 skill 注入时仍能执行相同的程序；
- 从 harness 强制安排 reflection，过渡到模型根据失败信号自主反思；
- 从 harness 生成完整计划，过渡到模型自己维护子目标和长程状态；
- 从自动 summary 和 retry，过渡到模型自己决定如何压缩信息和恢复错误。

最终目标是：这些能力在训练早期可以由外部 harness 扶持，但在训练后期逐渐内化为模型策略；部署时只需要保留轻量的工具、权限、sandbox、存储和验证基础设施，而不必继续依赖完整的语义控制脚手架。

这和单纯增加任务难度或 harness 数量不同。任务难度增加，主要迫使模型在现有支架下处理更长的任务；harness 多样性增加，主要迫使模型适应更多接口和控制结构；Harness Fading 则直接改变“有多少能力由外部支架承担”，并把能力归属从 harness 逐步推向模型。

因此，我们要研究的核心问题是：

> **能否把 harness 作为训练早期的外部教师和稳定器，再在训练过程中逐渐撤掉其语义支撑，使模型在最终的轻量 harness 下自主完成原本需要厚 harness 才能完成的长程任务？**

在这个视角下，harness、模型和任务环境不是彼此替代的关系，而是一个逐步迁移能力归属的协同系统：强 harness 帮助较弱模型进入复杂任务，训练把其中一部分能力沉淀到模型，变薄后的 harness 再把模型暴露给新的任务难度和新的环境变化。

后续需要进一步回答的，是如何定义 harness 的“厚度”、如何安全地逐步撤除组件，以及如何区分真正的能力内化和单纯的接口适应。这些属于后续的评测与实现问题。
