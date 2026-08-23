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
| `list_mcp_resources` | 1 | `codex-rs core` MCP | 完整迁移；启用时必须连接实际 MCP 资源。 |
| `list_mcp_resource_templates` | 1 | `codex-rs core` MCP | 完整迁移；启用时必须连接实际 MCP 资源。 |
| `get_goal` | 4 | `codex-rs ext/goal` | 完整迁移。 |
| `create_goal` | 4 | `codex-rs ext/goal` | 完整迁移。 |
| `update_goal` | 3 | `codex-rs ext/goal` | 完整迁移。 |
| `shell_command` | 1,336 | 旧 shell 协议；不是当前 `codex-rs` handler | 自实现，字段保持 `command`、`workdir` 等历史形式。 |
| `exec` | 3,669 | `codex-rs core` code-mode | 对齐 raw JavaScript 参数、cell 和嵌套工具调用语义。 |
| `wait` | 980 | `codex-rs core` code-mode | 对齐 `cell_id`、`yield_time_ms`、`max_tokens` 与 cell 生命周期。 |

因此，真正需要 nanoGPT 自行定义功能的只有 **1 个不同工具**：`shell_command`。它占 1,336 次调用，约占全部调用的 3.4%。`exec` 与 `wait` 虽由采集宿主暴露，但本地 `codex-rs` 已有 code-mode 的工具规范和运行时作为行为参考。

## `codex-rs` 参考范围

`codex-rs` 作为 17 个工具的行为参考和测试 oracle，不作为运行时依赖。

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
