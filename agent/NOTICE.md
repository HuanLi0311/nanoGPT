# Agent 轨迹合成规范

本文是轨迹合成的实现边界和验收标准。图谱只负责提出候选、组合模式和
审计证据；真实 sandbox 执行与独立 verifier 才决定轨迹是否可用。

## 1. 硬约束

- 格式正确不等于数据高质量。
- 每个 tool call/result 必须可解析，并通过 `tool_call_id` 正确关联。
- observation 必须来自真实或可审计的有状态执行，不能由模型臆造。
- 必须记录 `state_before -> action -> state_after` 和状态变化。
- 最终结果必须由独立 verifier 判定，不能使用模型或 agent 自报完成。
- harness、工具适配器、parser、超时或 verifier 故障不能作为模型标签。
- 普通正向 SFT 只收录最终 verifier 通过且 harness 健康的具体轨迹。

工具中途失败不自动淘汰：如果轨迹正确恢复并最终通过 verifier，可以作为成功
SFT；最终失败的轨迹不能作为成功 gold，可进入诊断或 RL 数据。

## 2. 两层图谱

### 概念/任务图

用于提出任务和覆盖目标，不表示真实执行事实：

```text
domain -> subdomain -> concept/material -> task_family
```

例如：

```text
<coding, has_subdomain, code_repair>
<code_repair, has_concept, test_failure>
<test_failure, synthesizes, repair_task_family>
```

### 执行/状态图

每次工具调用是事件节点，基本结构为：

```text
state_t --from--> event_t --to--> state_t+1
                         \--invokes--> tool
```

工具对多个实体的影响用多条边表示：

```text
<event:c1, reads, file:a>
<event:c1, modifies, file:b>
<event:c1, produces, file:b@v1>
<event:c1, returns, result:r1>
```

状态至少拆成四类组件：

- `environment_state`：文件、目录、进程、session、图像、数据库等；
- `agent_context_state`：observation、消息、记忆和当前计划；
- `harness_state`：工具集、pending call、call ID、协议和 turn budget；
- `goal_verifier_state`：任务目标、断言、verifier 版本和当前结果。

具体快照、参数、完整结果和 payload 保存在不可变 episode 证据中；图节点可以
只保存 hash 和结构化事实，但抽象边必须能回指具体 episode/event/verifier。

## 3. 抽象模式与组合

不要直接拼接互不共享环境的 raw message、shell 输出、session ID 或 call ID。
从成功执行中归纳可参数化动作模式：

```text
<action:patch, requires, pattern:source_has_bug>
<action:patch, produces, pattern:source_fixed>
```

只有满足以下条件才允许组合：

- 前一个动作的 effects 满足后一个动作的 preconditions；
- 工具 schema、参数类型、权限和 workspace 兼容；
- session 生命周期、状态版本和 verifier 兼容；
- 新状态在真实执行前只能标记为 `predicted`，不能作为 gold 或 reward。

图谱组合本身不能扩大状态空间。真正的新颖性来自新 workspace/初始状态、动作
参数、分支与恢复路径、新工具组合或新任务族。记录新颖性等级：

```text
known_replay < new_path < new_instance < new_state_pattern
             < new_tool_combination < new_task_family
```

## 4. 工具边界与领域

优先使用 `agent/runtime` 已有工具：

- 环境动作：`exec_command`、`write_stdin`、`apply_patch`、`view_image`；
- verifier：在工具执行后 out-of-band 调用，不作为模型自报结果；
- `spawn_agent`、`send_message`、`wait_agent`、`update_plan` 等属于 harness/协作
  图，不与 workspace 状态图混合；`shell_command` 是兼容别名；`exec`/`wait`
  是编排工具，不重复计为环境动作。

第一阶段覆盖以下子领域：

1. 文件与 workspace；
2. 代码检索与理解；
3. 代码修复、测试与构建；
4. 配置、数据和 CLI 流程；
5. 进程、日志和 session；
6. 图像及其他文件产物检查。

当前程序化 runner 只实现 `exec_command` 和 `apply_patch`；增加其他工具前必须
先实现真实 adapter、状态追踪和 verifier，不能只把工具名写入 manifest。

## 5. 合成 pipeline

1. **冻结契约**：固定 task schema、初始 workspace 生成器、工具 schema、harness、
   timeout/turn budget、verifier 版本和随机种子。
2. **收集种子**：使用有真实执行证据的 episode 或新建任务 manifest，记录目标、
   初始文件、动作模式和独立完成断言。
3. **归纳模式**：从成功 episode 提取 `state -> event -> state`，保留 concrete
   event 作为证据，不用抽象图替代原始轨迹。
4. **组合候选**：从概念图采样任务和约束，从执行图组合兼容动作模式，并参数化
   workspace、文件、配置、工具组合和恢复路径。
5. **真实执行**：每个候选在隔离 sandbox 中由 trace runner 逐步执行真实工具，
   不接真实模型作为 teacher。
6. **独立验证**：执行 verifier，记录 task outcome、状态变化、协议、harness 和
   failure class。
7. **环境门槛**：每个任务环境进行 100 次相互独立的程序化 rollout；只有至少
   一次通过 verifier，即 `pass@100 > 0`，才进入可训练 RL 任务池。不能复制同一
   条确定性轨迹冒充 100 次；固定轨迹只能记为 `replay_pass`。
8. **轨迹过滤**：只有协议有效、call/result 完整、trace fidelity、harness 健康、
   具体轨迹 verifier 通过的样本进入普通成功 SFT。
9. **保存与反馈**：成功轨迹回写图谱；健康 harness 下的失败轨迹用于诊断/RL；
   harness fault 或 unknown 进入回归集，不给模型错误正负标签。按状态模式、
   工具组合、恢复方式和 verifier 覆盖率决定下一轮采样。

## 6. 现有 Codex 数据的使用

`data/post_train/data/rendered/sft/codex_train.parquet` 和
`codex_test.parquet` 必须保留，不覆盖、不删除。

它们可用于：

- 挖掘真实 prompt、任务族、工具参数和工具组合；
- 提供动作模式、轨迹长度和难度分布的先验；
- 筛选出可重建 workspace/verifier 的任务并重新 sandbox 执行；
- 作为已有 SFT 基线和协议冷启动数据。

它们不能直接证明状态转移或任务成功：若缺少初始状态、状态快照和独立
verifier，只能标记为 `unverified_behavior`，不能直接写入成功状态图或正向
SFT。`tests_passed`、`exit_code_0`、`patch_success` 等信号只能用于筛选重放
优先级，不能替代 verifier。

新数据写入：

```text
model/language_model/data/post_train/data/raw/synthetic
model/language_model/data/post_train/data/jsonl/synth       # 确有需要时
model/language_model/data/post_train/data/rendered/sft       # 通过筛选的 Parquet
model/language_model/scripts/synthesis                       # 合成代码
```

现有 Codex 与 synthetic 数据按 assistant token 做配比实验，不按 episode 数量
硬拼；先做小规模 ablation，不设固定“正确比例”。

## 7. 每个 episode 的最低证据

至少保存：

```text
task_id, environment_id, initial_state_hash, seed
harness_version, tool_schema_version, verifier_version
actions, events, state_before, state_after, state_delta, tool_result
protocol_status, call_result_linkage_complete, trace_fidelity
task_success, independent_verifier_passed, harness_status, failure_class
```

验收时必须能回答：用了什么初始环境、调用了什么工具、状态如何变化、verifier
为何通过，以及该抽象边能回溯到哪条具体证据。

## 8. 禁止事项

- 不把长、复杂、格式正确的失败轨迹当成功 SFT；
- 不把模型自报完成、shell 输出或 heuristics 当独立 verifier；
- 不把抽象组合出的 `predicted` 状态当事实、gold 或 reward；
- 不跨 episode 拼接 raw observation、session/call ID 或工具结果；
- 不把 harness/tool adapter/verifier 故障标成模型失败；
- 不声称图谱覆盖全部可达状态空间，只报告已验证的状态模式和覆盖率。
