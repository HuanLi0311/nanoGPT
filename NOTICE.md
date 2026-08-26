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