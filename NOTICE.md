整理了一些 Kimi K3、Qwen、快手、DeepSeek 四家技术报告里对这件事的自陈与解法摆在一起看：接口过拟合究竟过拟合了什么、四家各在哪一层下手、以及哪一条已经有受控实验支撑。以及我们自己在业务上遇到的系列问题。

# Moonshot 的 K3 技术报告（arXiv 2607.24653）
在这条线上说得最像方法论、也最克制。§4.2「RL Task Synthesis and Agentic Environments」开篇第一句，就把多样性放到了前置条件的位置：我们 RL 框架的有效性高度依赖于丰富、多样、且可稳健验证的环境。为支撑跨复杂长程任务的可扩展训练，我们设计了一系列专用白盒环境与任务合成范式。

K3 的评测配置写在 §6，我认为是全篇最该被抄的一段：

除按 harness 拆行的 benchmark 外，Table 3 的 Harness 列给出的是 Kimi K3 所用的 harness。其他模型中，Claude 系列与 GLM-5.2 用 Claude Code 评测，GPT 系列用 Codex。例外是所有模型统一使用指定 harness 的那几个：24/7 ClawBench 2.0 用 OpenClaw；MIRA Bench 用 MIRA（一个内部的分布外 harness）；Agent Behavior Bench 与 Chat All-in-One 用 Kimi Work；CLIF 与 Agentic Vision Bench 用 Kimi Code。
三个细节值得单拎：

结果表里有一列专门叫 Harness。 报分带 harness 名，这件事在 K3 这里已经是表格结构的一部分。
专门留了一个 OOD harness 打自己。 MIRA 的定语就是 an internal out-of-distribution harness，MIRA Bench 上所有模型统一用它。K3 在这上面 64.1，落后 Claude Fable 5 的 72.9——他们把自己在 OOD harness 上打不过的事实印在了表里。
同一个 benchmark 同时报两个 harness。 官方 blog 脚注 8 写明 K3 同时在 Kimi Code 与 Claude Code 两个 harness 下评测，KCB 2.0 上 Claude Code 73.7、自家 Kimi Code 72.9，差 0.8 点。这是「我没有过拟合到自家壳」最有效的自证方式，比任何声明都管用。
第一条我认为是目前最便宜的「接口过拟合」探针，可以直接抄：训练时完全不用某个 harness，评测时所有模型统一用它。 配套的报分口径是——必须带 harness 名 + 版本号，跨 harness 不直接比。

1.2 但千万别把 K3 神化
它在协议这一层反而绑得更死。技术报告 Appendix 讲 chat template 的那段写得很直白：

Kimi K3 只支持 preserved thinking：在 thinking 模式下，think 通道始终保留在历史中——即使内容为空也保留——以便模型在各轮之间观察到一致的消息结构；在 instruct 模式下，历史消息只包含 response 与 tools 两个通道。
官方 blog 的 Limitations 第 1 条就是这件事的后果：

对 thinking 历史的敏感性。 K3 是在 preserved thinking history 模式下训练的。如果 agent harness 未按要求把全部历史 thinking 内容传回，或者把一个正在进行的其他模型会话切换给 K3，生成质量可能变得高度不稳定。我们建议使用经过兼容性验证的 harness，例如 Kimi Code，并避免在会话中途切换到 K3。看明白了吗——K3 把「任务 harness」随机化了，在 wire 协议这一层（thinking history 怎么回传）却要求 harness 严格配合。 任务级的环境随机化解决不了协议级的硬耦合，所以准确的说法是：K3 提供的是方法论，谈不上免疫力。


# Qwen：把「单一 tool chat template 会过拟合」写进报告，还做了受控消融
Qwen 的完整披露在2603.00729，2026-02-28。我认为这份报告在这个问题上的处理是几家里最完整的：问题陈述、机制、训练对策、受控消融、专门评测、RL 惩罚项，六件事全齐。

问题是从内部 benchmark 反馈里冒出来的（§4.2.2 User Experience Expert）：

我们观察到，标准的软件工程任务（例如修 GitHub issue）无法完全覆盖真实 CLI/IDE 场景下 agentic coding 的挑战。……特别是，我们发现不同的 CLI/IDE 脚手架（如 Cline、Qoder、OpenCode 等）采用了各不相同的 tool-calling schema，这对模型可靠地遵循 tool-call 格式构成了实质性挑战。然后是全篇我认为最该被引的一句，它把这件事的因果直接点明了：许多现有模型是用单一的 tool chat template 训练的，这往往导致过拟合到特定的输出结构，并在部署到未见过的 tool-calling 格式下时鲁棒性下降。 实际中，真实的 agent 系统使用五花八门的格式约定，用户也经常直接在 system prompt 里自定义 tool-call schema。为提升泛化，我们使用多样的 tool chat template 与格式来训练模型。

注意这句的主语是「许多现有模型」——Qwen 是在描述整个行业的默认做法，而这个默认做法正好就是「模型过拟合自家脚手架」的成因。

2.1 变化轴与具体取值
如 Figure 4 所示，tool chat template 在若干关键轴上存在差异，包括工具集定义、工具调用格式、以及工具返回的包装方式。
我们的训练数据纳入了广泛的工具表示形式，包括自然语言工具描述、JSON 格式、Python 风格调用、XML 风格 schema（含 qwen3_coder）、以及 TypeScript 风格接口。通过在训练中让模型接触多样的格式约定，模型学到的是与格式无关的工具使用行为，而不是记住某一种输出结构。
值得注意的是 qwen3_coder 这个 XML 风格格式本身就是 Qwen 自家的一个协议分叉（为了避免 JSON 对多行代码的转义开销），而他们把自家这个分叉也当成训练分布里的一个取值摆了进去。

他们做了模板数量的受控消融: 经验上，增加训练所用 tool call template 的数量，会持续提升下游对格式变化的鲁棒性。如 Figure 5 所示，即使数据量与训练配方保持不变，SWE-bench Verified 上的表现也随模板多样性的增加而提升。 这些结果表明，训练阶段的格式多样性是提升部署时对新 tool-calling 格式泛化能力的有效手段。Figure 5 的图题写得更明确：SWE-bench Verified 表现随 tool chat template 数量的变化；数据量与训练配置保持一致。

数据量固定、训练配置固定，只加模板数量，SWE-bench Verified 就涨。 这条把「接口多样性」从一个鲁棒性论证升级成了一个能提升主 benchmark 的手段——它跟其他几家「为了泛化牺牲一点峰值」的直觉是反的。

2.3 专门的跨脚手架评测
不同的 IDE/CLI 框架，例如 Qwen-Code、Trae、OpenCode、Cline、KiloCode，采用了各自定制的 prompt 模板，以及互不相同的 function-calling 与 MCP 交互格式。……这种多样性对单个模型跨社区常用 IDE/CLI 的泛化构成了显著挑战。
为系统评估这项关键能力，我们构建了一个内部评测集，显式衡量模型对不同真实 agentic coding 脚手架/IDE 的适应性。该评测集由取自代表性 IDE/CLI 脚手架的多套 prompt 模板与 tool-call schema 组成，覆盖 system instruction 的差异、以及 XML 变体 / JSON 两类 tool call 模式。
报告的 Table 2 给了五个 scaffold 上的准确率，并自陈各模型在不同 IDE/CLI 环境下的表现差异很大。

这张表的分数我不引。 它是厂商自建的 in-house benchmark、自家模型排第一，按本文一贯口径，这类数字不能用来做横向比较。可以引的是它的设计：一个专门衡量「同一个模型在五套社区脚手架的 prompt 模板与 tool-call schema 下能否稳定守格式」的评测集——这正是 §一 里 K3 那个 OOD harness 探针的加强版，从「留一个」变成「铺五个」。

2.4 RL 侧也挂了对应的惩罚项
其次，我们施加了 turn 级的 tool-format 惩罚。在每个交互步，我们用规则校验 tool-call 格式的正确性；优化时，与非法 tool call 关联的 token 会收到 token 级惩罚，以防模型学到畸形的工具调用模式。
接口正确性被直接写进了 reward，而不只是靠数据分布去覆盖。 这是 Qwen 跟其他几家的一个明显区别。


# 快手 KAT-Coder-V2.5：三类过拟合 + Harness Scaling

「接口依赖」四个字太笼统，不足以指导做事。目前最好用的分类法来自 KAT-Coder-V2.5 Technical Report（快手 KwaiKAT，arXiv 2607.05471，2026-07）。他们把这件事写进 Introduction 的三大挑战之一：长程 agentic 强化学习面临稀疏奖励、不稳定的环境反馈、粗粒度的信用分配、过拟合到某个固定 harness、以及难以融合各专精专家能力等问题。然后在 §4.1 Harness Scaling 里拆成三类：
Format overfitting Context-structure overfitting Control-flow overfitting 
我觉得这个倒是很好地把其他文章各自在讨论的观点和类型给汇总了一下, 比如文章里还说Harness Scaling 的关键不在于 harness 的数量，而在于多样性是否落在对泛化有用的维度上。Qwen 那三条基本落在快手第一条「工具调用协议」的内部，而快手多出了上下文管理与控制流两条。两家的覆盖面是嵌套关系——快手更宽，Qwen 在最窄的那一维上做到了有消融曲线。

# 一个反例：环境规模化 ≠ 接口随机化
智谱的 GLM-5 技术报告是个有意思的对照。它的 Agent 环境扩展做得非常扎实：跨数千仓库、覆盖九种语言、构建了 10000+ 个可验证环境。但我抓到的部分里没有任何关于 harness 随机化或工具协议多样化的表述，内部评测用的是「Claude Code 评估集合」。

这未必是缺陷——GLM-5 要解决的主要矛盾可能就不在这儿。但它说明一件事：「环境规模化」和「接口随机化」是两个正交的维度，而绝大多数团队只做了前者。 一万个可验证环境，如果都通过同一套工具 schema、同一种上下文管理策略暴露给模型，那模型见到的接口分布仍然只有一个点。

------------------------------------------------------------------

# 延伸一下 在后训练的过程当中去改harness  让harness逐渐由厚变薄 从而在早期稳定模型做简单推理 后期倒逼模型自己实现长程的能力

可以，而且我认为这是一个独立性很强的研究问题，但它不是和已有工作割裂的，而是建立在已有三类 harness 多样性之上的更高一层变量。

  一句话概括：

  > KAT-Coder 的三轴回答“harness 在哪里变化”；你的问题回答“完成能力最终由谁承担”。

  ## 三个问题要分开

   研究变量                   训练时改变什么                                             主要学到什么
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   任务/环境复杂度            文件、数据库、工具链、状态空间、任务长程性                 更长的规划、状态追踪和环境建模
  ─────────────────────────  ─────────────────────────────────────────────────────────  ───────────────────────────────────────
   Harness 多样性             Format、Context-structure、Control-flow                    对不同接口和脚手架的泛化
  ─────────────────────────  ─────────────────────────────────────────────────────────  ───────────────────────────────────────
   Harness 厚度/脚手架强度    planner、reflection、summary、skill、retry 等外部支持的    把原本由 harness 承担的能力迁移进模型
                              多少

  所以：

  - 固定 harness、增加任务难度：训练模型在同一脚手架下处理更长、更复杂的任务；
  - 固定 harness 厚度、换协议：训练模型适配不同接口；
  - 固定任务，逐渐削弱 harness：训练模型接管原本由 harness 完成的工作。

  这三者互相关联，但不是同一回事。

  ## 为什么“环境变复杂”不等于“harness 变薄”？

  假设 harness 每隔五步自动：

  1. 总结历史；
  2. 生成下一阶段计划；
  3. 调用 critic；
  4. 判断是否重试；
  5. 决定什么时候停止。

  即使任务从 10 步变成 1000 步，模型也许只是持续执行 harness 给出的局部安排。它会产生更长的 trajectory，但不一定真正学会了长程
  规划。

  相反，如果把 planner、reflection 和自动 retry 拿掉，模型必须自己：

  - 保存长期目标；
  - 判断什么时候重新规划；
  - 发现错误；
  - 决定继续还是停止；
  - 组织下一阶段行动。

  这才是在训练“能力归属发生迁移”。

  因此，“更长的 rollout”不一定意味着模型内化了长程能力；有可能只是 harness 把长程控制外包掉了。

  ## 你说的 thick-to-thin 是可行的

  我会把它叫作：

  - scaffold annealing；
  - harness fading；
  - harness curriculum；
  - external-to-internal capability transfer。

  基本过程可以是：

  弱模型 + 厚 harness
          ↓
  稳定解决复杂但可验证的任务
          ↓
  逐步移除 planner / skill injection / forced reflection / context assistance
          ↓
  模型接管这些决策
          ↓
  薄 harness + 更强模型
          ↓
  进入更复杂任务和新环境

  这和 teacher forcing 或课程学习有点相似：早期外部脚手架承担一部分困难，后期逐渐撤掉。

  不过不建议简单地“整个 harness 一起变薄”。更合理的是按组件逐个退坡：

  1. 先移除显式提示，例如“现在请反思”；
  2. 再让 reflection 从强制变成可选；
  3. 再移除自动 planner；
  4. 再削弱 context summary；
  5. 最后测试模型能否在基本工具接口下独立完成任务。

  ## 哪些部分应该变薄，哪些部分不应移除？

  可以把 harness 拆成三层。

  ### 不应随意移除的硬基础设施

  - 工具序列化和 token 协议；
  - 权限控制；
  - sandbox 隔离；
  - 文件和数据库访问；
  - 独立 verifier；
  - 外部 API；
  - 并行计算和持久化存储。

  这些不是“模型应该内化的智能”，而是运行环境本身。

  ### 可以逐渐变薄的语义脚手架

  - system prompt 中的任务分解；
  - 自动 planner；
  - 强制 reflection；
  - 自动总结和上下文重组；
  - skill 内容注入；
  - 自动 retry；
  - 停止判断；
  - 子 agent 分工；
  - critic 或 judge 的显式反馈。

  ### 不一定要变薄的外部资源

  比如向量数据库、长期 memory、实时文件系统。模型可以内化：

  - 什么时候检索；
  - 如何构造 query；
  - 如何判断结果不足；
  - 如何保存和恢复状态。

  但不应该假设模型会内化数据库中持续变化的事实。内化的是“使用外部资源的策略”，不是资源本身。


  因此，在 NOTICE.md 讨论的范围内，你这个方向完全可以独立成一个主题。

  ## 最重要的实验不是“分数变高”，而是 scaffold gap 变小

  可以定义一个简单指标：

  scaffold gap
  = Performance(thick harness)
    - Performance(thin harness)

  如果能力真正内化，随着后训练推进：

  - 厚 harness 下的性能继续提高；
  - 薄 harness 下的性能也提高；
  - 两者之间的差距缩小；
  - 在未见过的 harness 上仍能保持能力。

  因此应至少比较：

  1. 固定厚 harness；
  2. 固定薄 harness；
  3. 厚薄随机混合；
  4. 厚到薄的阶段性 curriculum；
  5. 只改变协议但不改变厚度；
  6. 只改变任务难度但不改变 harness；
  7. 未见过的 harness。

  这样才能区分：

  - 模型学会了更复杂的任务；
  - 模型学会了适配更多协议；
  - 模型真的接管了原本由 harness 完成的规划、反思和状态管理。

  ## 最后的判断

  你的想法不是简单地给 Format、Context、Control-flow 再加一条轴。

  更准确的结构是：

  外部多样性空间：
  Format × Context-structure × Control-flow × Task-world environment

  能力归属元维度：
  externalized scaffold  ←→  internalized model policy

  前三者描述“哪里变化”，后者描述“能力是否从外部迁移到模型内部”。

  所以未来更强的 harness 不一定意味着更厚。比较理想的状态是：

  > harness 在协议、执行、安全、验证这些基础层保持稳定；在规划、反思、记忆、路由和控制流这些语义层可以逐渐变薄，同时持续引入新
  > 的任务环境和未见过的 harness，检查模型是否真的获得了自主能力。




可以形式化成：

  H_t = H_core + S_t

  其中：

  - H_core：始终保留的基础设施，例如工具协议、权限、sandbox、文件/数据库访问、独立 verifier；
  - S_t：可逐渐撤掉的语义脚手架，例如 planner、reflection、skill 注入、自动 summary、retry、subagent 调度等。

  训练初期：

  S_t 很厚，模型能力不足时由 harness 补足

  训练后期：

  S_t 逐渐被 mask / 删除，模型必须自己完成规划、反思、记忆和控制

  部署时：

  H_deploy ≈ H_core

  ## 这和任务难度、harness 多样性不是一回事

  可以把它们分成三个训练变量：

   变化                        主要训练什么
  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   任务环境复杂度增加          长程推理、状态追踪、环境建模
  ──────────────────────────  ─────────────────────────────────────
   Harness 形式/结构多样化     跨格式、跨上下文、跨控制流泛化
  ──────────────────────────  ─────────────────────────────────────
   Harness 语义支持逐渐减少    原本由 harness 承担的能力内化到模型

  所以你说的三种情况分别是：

  固定 harness + 更复杂环境
      → 学会在这个脚手架下处理更长任务

  同样厚度 + 换一种协议或模板
      → 学会适配新的 harness 表面

  同样任务 + 删除 planner / reflection / skill 等组件
      → 学会接管这些组件原本承担的能力

  第三种就是你提出的独立问题。它和 KAT-Coder 的 Control-flow overfitting 有交集，但重点不同：

  - KAT-Coder 主要问：模型是否见过不同的控制流；
  - 你的问题问：模型是否能够逐步摆脱外部控制流。

  前者是 harness diversity，后者是 scaffold fading / capability internalization。

  ## 但“删除组件”不一定自动导致内化

  这里有一个关键风险：如果 harness 直接替模型完成了决策，而模型没有被训练去重建这些决策，那么后期删掉组件时可能只是性能崩溃，并
  不是能力内化。

  例如：

  - planner 给了完整计划，模型只负责执行；
  - reflection 模块直接告诉模型哪里错了；
  - skill 直接注入了完整行为规范；
  - context manager 直接保留了模型本来会丢失的关键信息。

  这时模型可能学到的是“消费外部答案”，而不是“自己产生答案”。

  因此早期 harness 最好提供的是：

  - 部分计划，而不是完整解法；
  - 反思信号，而不是直接修复方案；
  - 任务边界和检查标准，而不是最终结论；
  - 可学习的轨迹和反馈，而不是隐藏的 privileged state。

  同样可以对 planner、自动 summary、critic、retry 和 subagent delegation 做渐进撤除。

  ## 我建议不要做单调的“一刀切变薄”

  更稳妥的是：

  早期：厚 harness
  中期：厚薄 harness 混合采样
  后期：薄 harness 为主，厚 harness 少量 replay

  并且按组件独立退坡。例如先去掉强制 reflection，再去掉 planner，最后才削弱上下文辅助。

  最好根据模型表现自适应退坡：

  - 协议错误多：暂时保留格式适配器；
  - 状态漂移多：暂时保留 summary；
  - 不会重试：暂时保留错误反馈；
  - 已经能自主完成：撤掉对应组件。

  也就是说，变薄不一定按照训练 step 机械发生，而可以由“模型是否已经接管该功能”决定。

  ## 哪些组件最终不适合删除？

  轻量 harness 不等于完全没有外部系统。

  通常可以内化的是：

  - 规划策略；
  - 反思时机；
  - 任务分解；
  - skill 使用判断；
  - 失败恢复；
  - 停止判断；
  - 检索策略；
  - 状态压缩策略。

  但以下内容仍然适合留在外部：

  - 实时数据库和向量数据库；
  - 最新文件和外部 API；
  - sandbox 执行；
  - 权限与安全检查；
  - 独立 verifier；
  - 大规模并行 subagent；
  - 超长上下文存储。

  模型可以内化“如何使用向量库”，但不能替代一个持续变化的向量库；可以内化“如何验证代码”，但不应完全替代真实测试器。

  ## 这个方向应该如何证明有效？

  需要比较：

  1. 始终使用厚 harness；
  2. 始终使用薄 harness；
  3. 厚薄随机混合；
  4. 厚到薄的 curriculum；
  5. 厚到薄但不做蒸馏；
  6. 厚到薄并做中间能力蒸馏。

  评估时同时测：

  - 最终薄 harness 上的成功率；
  - 未见过 harness 上的成功率；
  - 删除各个组件后的性能下降；
  - 工具调用错误率；
  - 幻觉、提前停止和恢复能力；
  - 成本、步数和延迟。

  尤其要看：

  internalization gap
  = Performance(thick harness)
    - Performance(thin harness)

  如果训练有效，后期应该看到薄 harness 性能上升，而且这个 gap 缩小。