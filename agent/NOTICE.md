# Agent 工具同步：范围、来源与验收边界

## 结论

这次同步的目标不是缩减工具集，也不是把完整 `codex-rs` 产品搬入训练环路。

目标是完整覆盖当前 `filtered/*.jsonl` 中实际出现的 **20 个不同工具名**（4,594 条样本、39,785 次调用）：模型训练、推理和轨迹回放继续使用原有的工具名、参数字段、结果字段和 `tool_call_id` 关联方式，不新增第二套模型可见协议。

| 分类 | 工具数 | 调用次数 | 处理原则 |
| --- | ---: | ---: | --- |
| `codex-rs` 有 handler、extension 或 code-mode reference | 19 | 38,449 | 对齐当前轨迹所用的语义、功能、状态和生命周期。 |
| `codex-rs` 没有同名当前 handler | 1 | 1,336 | 保全原有字段和结果关联；根据轨迹用途实现实际功能。 |
| 合计 | **20** | **39,785** | 不删除、合并或额外增加模型可见工具。 |

这里的“对齐”限于当前数据和 reference 已暴露的外部可观察行为：同名调用产生同类作用、相同状态转移、成功/失败边界和相应结果。它不要求采用 Rust、FFI、守护进程或 `codex-rs` 的内部架构；内部实现选择以简单、可复现、可测试为准。未出现在轨迹中的远程账号、权限审批、云端 MCP 发现等产品能力不在本轮范围内。

对于没有同名当前 handler 的工具，字段和调用关联是硬约束；具体功能由 nanoGPT harness 自己定义。只要状态、结果和错误对模型可见且稳定，模型可以通过 SFT 与 RL 学会这套新功能及其生命周期。不能把它们做成空操作或只改名转发。

任务 verifier 是运行时的独立评估步骤，不增加为第 21 个模型工具。它负责判断任务是否完成和区分 harness 故障，不能被“工具调用格式正确”替代。

## 当前实现门槛（2026-08-24）

独立审查确认，本文件所述目标尚未由当前 verl 启动路径实现，因此不能清空本文件或转入知识图谱数据扩充：

1. `verl_grpo.sh` 现有 YAML 只加载 `exec_command`、`apply_patch`、`verify_task` 三个旧工具；字段也与轨迹不一致，并错误地把独立 verifier 暴露给模型。启动器已默认拒绝该配置。`ALLOW_LEGACY_VERL_TOOLS=1` 只保留给隔离的旧 workspace smoke，不是训练开关。
2. verl 的标准 loop 对每次调用都 create/release 一个 Python tool；它不会保留 shell、cell、goal 或子 agent 的 episode 状态。当前的 20 工具 TypeScript runtime 没有进入该 loop。
3. Qwen3 默认 chat template 会在渲染时丢弃 assistant call ID 和 tool result ID；对于 raw `apply_patch` 又会生成标准 JSON parser 无法读回的文本。Parquet 中虽保留字段，实际 SFT/rollout token 序列仍不对齐。
4. 轨迹中的 `apply_patch` 是 raw patch 字符串，`exec` 是 raw JavaScript 字符串；其余工具是 JSON 字符串。不能把前两者改成 `{patch: ...}`、`{source: ...}` 之类的新模型可见参数再称为语义对齐。
5. Qwen SFT 当前默认从 `Qwen3-8B-Base` 开始，而旧 GRPO 默认是 `Qwen2.5-Coder-7B-Instruct`。两阶段必须指向同一个 student checkpoint；默认值现已统一为 Qwen3 SFT 产物。

下一阶段应提供一个能同时满足以下外部行为的 runtime bridge：训练模板和 rollout 都可表达并回传同一 `tool_call_id`；解析层保留 raw patch/JavaScript 输入；一个 episode 内的 20 工具共享真实状态；任务结束后独立执行 verifier，并把 verifier、工具、协议和 harness 故障分开写入 reward。实现可以复用现有 TypeScript runtime 或等价地移植其行为，但不能只扩写 YAML。

## 20 个工具的来源与边界

| 工具 | 调用次数 | 来源 | 本次要求 |
| --- | ---: | --- | --- |
| `exec_command` | 23,169 | `codex-rs core` unified exec | 完整迁移。 |
| `write_stdin` | 8,360 | `codex-rs core` unified exec | 完整迁移，包括与 shell 会话的关联。 |
| `apply_patch` | 1,783 | `codex-rs core` | 完整迁移，保留历史 freeform patch 输入。 |
| `view_image` | 145 | `codex-rs core` | 完整迁移；模型输入链路必须能真正接收图像结果才算完成。 |
| `update_plan` | 138 | `codex-rs core` | 完整迁移。 |
| `request_user_input` | 94 | `codex-rs core` | 完整迁移；运行环境需要提供真实或任务定义的用户响应。 |
| `wait_agent` | 65 | `codex-rs core` multi-agent | 完整迁移。 |
| `send_message` | 17 | `codex-rs core` multi-agent | 完整迁移。 |
| `list_agents` | 6 | `codex-rs core` multi-agent | 完整迁移。 |
| `spawn_agent` | 4 | `codex-rs core` multi-agent | 完整迁移。 |
| `followup_task` | 4 | `codex-rs core` multi-agent | 完整迁移。 |
| `interrupt_agent` | 2 | `codex-rs core` multi-agent | 完整迁移。 |
| `list_mcp_resources` | 1 | `codex-rs core` MCP | 对齐资源字段；未配置时返回空列表。 |
| `list_mcp_resource_templates` | 1 | `codex-rs core` MCP | 对齐模板字段；未配置时返回空列表。 |
| `get_goal` | 4 | `codex-rs ext/goal` | 完整迁移。 |
| `create_goal` | 4 | `codex-rs ext/goal` | 完整迁移。 |
| `update_goal` | 3 | `codex-rs ext/goal` | 完整迁移。 |
| `shell_command` | 1,336 | 旧 shell 协议；不是当前 `codex-rs` handler | 自实现，字段保持 `command`、`workdir` 等历史形式。 |
| `exec` | 3,669 | `codex-rs core` code-mode | 对齐 raw JavaScript 参数、cell 和嵌套工具调用语义。 |
| `wait` | 980 | `codex-rs core` code-mode | 对齐 `cell_id`、`yield_time_ms`、`max_tokens` 与 cell 生命周期。 |

因此，真正需要 nanoGPT 自行定义功能的只有 **1 个不同工具**：`shell_command`。它占 1,336 次调用，约占全部调用的 3.4%。`exec` 与 `wait` 虽由采集宿主暴露，但本地 `codex-rs` 已有 code-mode 的工具规范和运行时作为行为参考。

## `codex-rs` 参考范围

`codex-rs` 作为 19 个工具的行为参考和测试 oracle，不作为运行时依赖。

主要参考位置：

- shell/session：`../../harness/codex-rs/core/src/tools/handlers/shell_spec.rs`、`unified_exec/exec_command.rs`、`unified_exec/write_stdin.rs`；
- patch：`core/src/tools/handlers/apply_patch_spec.rs`、`apply_patch.rs`；
- 图像、计划、交互、MCP、多 agent：`core/src/tools/handlers/` 下的对应 handler/spec；
- code-mode：`core/src/tools/code_mode/execute_spec.rs`、`execute_handler.rs`、`wait_spec.rs`、`wait_handler.rs`；
- goal：`../../harness/codex-rs/ext/goal/src/spec.rs` 与 `tool.rs`。

迁移时必须覆盖工具在 reference 中已经暴露的输入、输出、状态、失败行为和相关联资源。例如：

- `exec_command` 与 `write_stdin` 的长命令/会话关系不能退化成两个无关的命令；
- `spawn_agent`、`wait_agent`、`send_message`、`list_agents`、`interrupt_agent` 必须操作同一组实际子 agent；
- `request_user_input` 不能只返回固定字符串；
- 未配置 MCP 资源时返回轨迹中的空列表；配置资源后返回相同的资源字段和服务端筛选结果；
- `view_image` 的结果必须进入模型上下文，而不只是落成一段无意义的文件路径。

这些是效果约束，不规定在 `agent/` 内使用何种类、进程模型、并发策略或代码组织实现。

## 历史工具与 code-mode

### `shell_command`

它来自较早的 shell 工具协议。当前 `codex-rs` 仍在模型配置中兼容 `shell_command` 这个名称，并将其视为 unified-exec 的历史别名，但没有对应的当前 handler。历史轨迹中它的输入是类似：

```json
{"command":"bash -n script.sh","workdir":"/workspace"}
```

nanoGPT 需要保留这组字段并实现“在指定工作目录执行 shell 命令、返回执行结果”的实际功能。内部可以复用与 `exec_command` 相同的底层能力，但模型可见的名称和字段不应改写成另一套协议。

### `exec` 与 `wait`

它们不是 shell 工具，但当前 `codex-rs` 已在 code-mode 中定义了相同的 raw JavaScript `exec`、cell ID 和 `wait` 协议。轨迹中的参数是 JavaScript 执行单元，例如调用 `tools.exec_command(...)` 后用 `text(...)` 写出输出；`wait` 使用同一 cell 的标识、等待时间和输出上限。

nanoGPT 中应保留这类 raw JavaScript 参数，提供一个实际可运行的异步 cell：cell 可以调用已注册工具并产生输出；运行中的 cell 必须有可供后续 `wait` 使用的标识。具体执行隔离和内部调度方式不在本文件限制，但不能把 JavaScript 仅当文本回显。

历史 `wait` 参数示例：

```json
{"cell_id":"22","yield_time_ms":30000,"max_tokens":60000}
```

nanoGPT 中的 `wait` 应对指定 cell 等待或轮询，并返回该 cell 新产生的输出或终态。它不是 `wait_agent`，也不应被改写为 shell 的 sleep 命令。

## 运行时改动边界

为使上述 20 个工具真正可训练和可评估，运行时需要保证以下效果：

1. 同一工具调用的 `tool_call_id` 从模型动作、执行、结果到轨迹保存保持一致；不额外发明模型可见 ID。
2. 所有有状态工具的状态可持续到后续调用：shell session、执行 cell、goal、子 agent、用户交互和 MCP 上下文不能在每一步被悄然重置。
3. 结果必须保留模型作下一步决策所需的信息，而不是只压缩成成功/失败布尔值。
4. 正常工具失败、参数/协议失败、外部服务失败、verifier 失败和 harness 故障必须分别记录；只有独立 verifier 证明任务成功时才给正向任务 reward。
5. 原始轨迹继续原样保存；训练/推理使用的渲染不引入新的前缀字段、工具别名或第二套参数格式。

本次不预先规定具体的 TypeScript 文件拆分、Rust 复用方式、进程/线程模型、并发调度规则或内部状态存储。只要上面的外部行为满足，选择最小且稳定的实现即可。

## 数据注意项

原始 artifact 是证据，应保留。当前数据中有一个已发现的协议异常：4 次 `get_goal` 调用中有 2 次带有 `exec_command` 风格参数，而 `codex-rs/ext/goal` 对 `get_goal` 的标准参数是空对象。这不构成第 21 个工具，也不应据此扩张 `get_goal` 的正式语义；后续回放时单独标记和审计该异常即可。

## 完成判据

同步完成的最低条件是：

1. 20 个工具名均可被运行时识别并产生真实结果；
2. 19 个有 `codex-rs` reference 的工具在代表性轨迹上满足其已暴露的外部行为；
3. `shell_command` 能完成其历史轨迹所表达的功能，`exec` 与 `wait` 能完成 code-mode cell 的执行、等待和结果传递，而非仅通过格式校验；
4. 完整轨迹可保留调用/结果关联并被重新渲染；
5. harness 故障不会伪装为模型负 reward，独立 verifier 的结果可用于教师模型与小模型的成对比较。

达到这些条件后，再扩充数据或做 RL，才能把失败更可靠地归因到模型、任务或 harness，而不是当前协议缺口。



----------------------------------------------------------------


独立审查时发现问题：

已完成的前置修复：

- SFT 文本投影保留 assistant call ID 与 tool result ID，并让三个数据入口复用同一序列化规则：model/language_model/scripts/tool_message.py:1。

- runtime verifier 现在验证调用和结果按 tool_call_id 一一对应，重复或错配 ID 会作为协议故障而非模型失败：agent/runtime/src/reward.ts:4。

- GRPO 默认拒绝旧三工具 YAML，避免把未对齐的 harness 当作 RL 环境；ALLOW_LEGACY_VERL_TOOLS=1 仅允许隔离 smoke：model/language_model/scripts/verl_grpo.sh:1。

- SFT 与 GRPO 默认 student 已统一为本地 Qwen3-8B 的 SFT 产物：model/language_model/config/verl_grpo.yaml:1。

验证通过：SFT 格式测试、trajectory artifact 自检、Node harness 自检、shell/YAML 语法检查，以及 launcher 的拒绝路径。期间一次 verl --help 实际进入了初始化，已终止，未留下训练进程、checkpoint 或 Ray 输出。

下一阶段是 runtime bridge，而不是加更多 YAML：需要自定义 Qwen 模板与 parser 保留 ID/raw 参数，并将 20 工具的共享 episode 状态和独立 verifier 接入 verl agent loop。完成这一层后再做知识图谱扩充，才能让新增轨迹的 reward 真正反映模型而非 harness 缺口。



-----------------------------------------------------------------------

## 知识图谱驱动的数据扩充

### 初衷

这里的图谱不是为了把 agent 轨迹画成更复杂的可视化，也不是把历史对话做通用知识库。它服务于一个更具体的问题：agent 的最终效果同时取决于基模和 harness；当工具协议、状态或 verifier 有缺口时，低 reward 不能证明模型能力差。若直接把这类轨迹拿去做 RL，模型会被错误信号推向格式投机、过早停止或规避工具。

因此，先用更强的 teacher 在同一任务和同一 harness 中作探针：teacher 能稳定完成、student 不能完成，才是可归因于 student 能力的候选样本；teacher 也在已知可行的任务上失败，或失败原因集中在工具/协议/verifier，则优先修 harness。这个过程把“模型问题”和“运行时问题”显式分开，并为后续高质量轨迹提供可信来源。

图谱的作用是把已验证轨迹中可复用的**任务意图、环境状态、工具动作、状态变化与验证结果**连接起来。相比按文本相似度检索或直接拼接 message，它保留了“为什么要调用这个工具、它依赖什么状态、它改变了什么、结果是否真正完成任务”的结构。这样扩充的是可执行任务分布和可靠的解题模式，而不是表面上更长的对话。

### 建图前提与数据边界

只有在本文件的运行时完成判据满足后才启动建图：20 个工具在同一 episode 中保持真实状态，`tool_call_id` 可从调用追到结果，raw patch/JavaScript 不被改写，独立 verifier 能区分任务失败与 harness 故障。原始 JSONL 和 artifact 继续作为不可变证据；图谱、标注和生成数据都是可再生的派生物。

每个 episode 至少应离线保留以下信息：任务输入、workspace 初始快照或可重建描述、按序事件、调用参数与结果、`tool_call_id`、可观察的文件/进程/会话状态变化、最终 verifier 输出、harness 版本和故障分类。时间戳、payload 等已有信息可以保留作诊断，但不应为了图谱而向模型训练或推理协议新增字段。

### 图谱结构

图以 episode 为根，事件仍按原始顺序保存；图中的节点和边只是对事件的离线索引与归纳：

- 节点包括任务意图、初始环境实体（文件、目录、命令、图像、会话、cell、goal）、观察结果、工具调用、工具结果、状态断言、最终答案、verifier outcome、harness 版本与失败类别。
- 顺序边描述事件先后；调用-结果边以原有 `tool_call_id` 连接；读取、产生、修改、等待、依赖和验证边描述对环境实体的关系。
- 任务成功、任务断言失败、正常工具失败、协议错误、外部服务错误和 harness 故障是不同的 outcome 节点/边，不能压成单个二元 reward。
- 跨 episode 只连接经规范化后确实相同或等价的意图、工具语义、文件模式和 verifier 模式；原始 message、call ID、session ID 不跨任务合并。

意图、实体和状态断言可由 teacher 离线抽取，但必须能回指到具体 event 或 verifier 证据。无法验证的抽取只作为检索提示，不能作为奖励、gold trajectory 或任务事实。图谱亦不取代原始轨迹：任何训练样本都必须能从图回溯到原始输入和一次真实执行。

### 从图谱到新数据

数据扩充按“图谱提出候选，真实执行决定收录”进行：

1. 从成功 episode 中抽取最小任务子图：一个意图、必要初始状态、工具依赖链和可执行的完成断言。按工具组合、调用深度、环境类型、失败恢复方式和任务领域分桶，补足低覆盖区域，而不是只采样高频 `exec_command`。
2. 只组合状态接口兼容的子图。例如“读取配置 -> 修改文件 -> 运行测试 -> verifier”可变为新的 workspace 任务；不能把来自两个无共享环境的 shell 输出、session ID 或子 agent 消息直接拼成一条轨迹。
3. 将候选子图实例化为隔离 workspace、初始文件和独立 verifier。任务必须可重复初始化、可执行、可判定；缺少 verifier 的候选只保留作 SFT/诊断候选，不进入 RL。
4. 让强 teacher 在目标 harness 中完成 rollout，并保存完整事件和状态转移。对同一任务可生成多个候选，以覆盖不同但有效的工具路径；去重依据是图结构和最终状态，不是只比较自然语言。
5. 重放或重新执行候选，使用独立 verifier、协议检查和 harness 健康检查过滤。只有任务成功、关联完整且 harness 健康的轨迹进入 SFT 或 RL；正常失败可用于困难度/恢复分析，harness 故障仅用于修复和回归测试。
6. 将通过样本回写图谱，更新各意图-工具-状态模式的成功率、teacher/student 差距和覆盖度。下一轮优先选择“teacher 可稳定解、student 尚弱、harness 健康且验证明确”的子图生成任务。

这形成一个受控闭环：修复 harness -> teacher 校准任务 -> 构建可信事件图 -> 生成并验证新任务/轨迹 -> SFT 与 RL -> 用更新后的 student 找到新的能力缺口。图谱不直接生成 reward，也不把失败伪装成监督；它把数据选择、任务组合和错误归因变成可审计的过程。

### 明确禁止的扩充方式

- 不跨任务拼接原始 assistant/tool 文本来制造“长轨迹”；这会破坏环境因果关系和 `tool_call_id` 生命周期。
- 不因相似 embedding 或相同工具名就认定两个状态可组合；必须满足输入/输出状态与 verifier 的兼容性。
- 不把 teacher 的文字判断当成功标签；执行后的独立 verifier 是唯一正向任务 reward 依据。
- 不用未对齐 harness 中的大量低分失败轨迹反向训练 student；先将其归类为模型失败、任务歧义、工具失败或 harness 故障。


