## 知识图谱驱动的数据扩充

这份说明定义 agent 轨迹扩充、质量判定和训练数据收录的边界。图谱是
离线的数据生成与审计工具，不是模型推理协议，也不能替代真实执行和
独立 verifier。

### 核心结论

SFT 数据的质量不等于 JSON 或 tool call 格式正确。质量至少包含：

- 协议可解析，调用和结果可以通过 `tool_call_id` 正确关联；
- 工具结果来自真实或可审计的有状态执行，不能是模型臆造的 observation；
- `state_before -> action -> state_after` 的状态变化一致；
- 最终任务由独立 verifier 判定完成；
- harness 健康，不能把 parser、工具适配器、超时或 verifier 故障当成模型标签。

长轨迹、工具种类多和格式正确只能说明覆盖或协议质量较好，不能证明任务
成功。普通正向 SFT 只收录最终 verifier 通过的轨迹。中间出现工具失败不
自动淘汰：如果模型识别失败、选择可行恢复路径并最终通过 verifier，该轨迹
仍然是成功轨迹；如果最终任务失败，则不把整条轨迹当作成功 gold。

### 两类图谱，不要混用

知识图谱扩充有两个互补层次：

1. **概念/任务图**：借鉴 Kimi K3，连接领域、细粒度概念、材料、任务类型
   和覆盖标签。它用于提出新的领域、任务、工具组合和约束，扩大任务来源，
   但不保证任务可执行。
2. **执行/状态图**：记录真实 episode 中的任务意图、初始状态、工具调用、
   observation、状态变化、恢复路径和 verifier 结果。它用于检查前置条件、
   组合可执行模式和追溯训练证据。

概念图可以表示为：

```text
<CS/AI, has_subconcept, GPU-kernel>
<GPU-kernel, related_to, Triton>
<Triton, supported_by, source-material>
<source-material, synthesizes, task-family>
```

这类关系表达知识和任务来源，不表达工具调用后的真实环境状态。

执行图的高层形式是：

```text
state_t --from--> transition_t --to--> state_t+1
                           \--action--> tool_call_t
```

其中一次工具调用是一个事件节点，而不是把所有工具塞进一个三元组字段：

```text
<call:c1, invokes, tool:read_file>
<call:c1, has_argument, path:calc.py>
<call:c1, reads, file:calc.py@v0>
<call:c1, returns, result:r1>
<call:c1, causes, observation:o1>
```

状态由多个组件共同决定时，状态节点连接组件节点；不要复制一个巨大的
嵌套状态作为每条边的 object：

```text
<state:s0, contains, file:calc.py@v0>
<state:s0, contains, process:pytest@p0>
<state:s0, contains, session:shell-1>
<state:s0, contains, goal:repair-add>
```

至少区分以下状态层：

- `environment_state`：文件、目录、进程、数据库、图像、session 等；
- `agent_context_state`：已观察到的结果、消息、记忆和当前计划；
- `harness_state`：pending call、`tool_call_id`、可用工具、turn budget 和协议状态；
- `goal_verifier_state`：目标、完成断言和当前验证结果。

工具可以读取或修改多个实体，使用多条边表示：

```text
<call:c2, reads, file:a>
<call:c2, reads, file:b>
<call:c2, modifies, file:c>
<call:c2, produces, file:c@v1>
```

只读工具可能不改变环境状态，但会改变 agent 的 observation/context。并行
工具调用使用 `parallel_group` 和 join 节点，不强行伪造成串行轨迹。原始
参数、完整结果和 payload 保留为不可变证据；图中的抽象边必须能回指到
具体 episode、event 和 verifier 证据。

### 抽象状态转移与状态空间边界

如果三元组中的 `S` 只表示已经观测过的具体快照，那么拼接三元组只能
重组已有 support，不能真正探索新的状态空间。因此执行图必须同时保存
可参数化的动作模式：

```text
<action:patch-file, requires, pattern:source-has-bug>
<action:patch-file, produces, pattern:source-fixed>
```

`pattern` 是由多个状态谓词组成的抽象状态模式，例如：

```text
<pattern:p1, contains, file-exists>
<pattern:p1, contains, test-fails>
<pattern:p2, contains, source-fixed>
```

只有当上一个动作的 effect 能满足下一个动作的 precondition，并且工具权限、
workspace、版本、session 和 verifier 依赖兼容时，才允许组合动作模式。
组合出的新状态在真实执行前只能标为 `predicted`，不能作为 gold trajectory、
任务事实或 reward。

扩充的新颖性分级如下，不能把“新文本”误称为“新状态”：

- `known_replay`：复现已有路径；
- `new_path`：已知状态和动作的新顺序；
- `new_instance`：同一动作模式的新参数、文件或 workspace；
- `new_state_pattern`：未覆盖的抽象前置/后置状态；
- `new_tool_combination`：新的工具依赖或并行组合；
- `new_task_family`：新的概念、目标或 verifier 类型。

真正的状态空间扩展必须使用至少一种外部生成或探索机制：

- 新建或扰动初始 workspace，例如注入不同 bug、配置、权限、进程和历史；
- 对抽象动作模式做参数化实例化；
- 从状态快照分叉，实际执行不同工具、参数和恢复路径；
- 让 scripted oracle、程序化轨迹生成器或探索策略执行新任务，而不只是重排旧消息；
- 由概念图引入未覆盖的领域、任务约束和工具组合；
- 按状态模式、工具组合、恢复方式和 verifier 覆盖率主动采样。

有限轨迹不可能穷举真实可达状态空间。目标是扩大可验证的抽象状态覆盖，
并用 held-out 初始状态、工具组合和任务族检查泛化，而不是声称图谱已经
覆盖全部可达空间。

### 标准扩充 pipeline

数据扩充按“图谱提出候选，环境真实执行决定收录”进行：

1. **冻结契约**：固定 task schema、初始 workspace 生成器、工具 schema、
   harness 配置、超时/turn budget、verifier 版本和随机种子策略。原始 JSONL、
   workspace artifact 和执行日志不可变保存。
2. **收集种子**：只从有真实执行证据的成功 episode 中抽取最小执行子图，
   包含意图、状态模式、工具依赖、恢复方式和完成断言。
3. **归纳模式**：把具体 `state -> call -> state` 归纳为带 precondition/effect
   的抽象动作；保留 concrete event graph 作为证据，不用抽象结果替代原始轨迹。
4. **提出候选**：从概念图采样新领域/材料/任务族，从执行图采样兼容的动作
   模式，并用状态生成器、参数化和环境扰动创建新的 workspace。禁止只拼接
   不共享环境的 shell 输出、session ID、raw message 或子 agent 文本。
5. **组合检查**：检查每个动作的输入/输出类型、前置/后置条件、工具权限、
   session 生命周期、状态版本和 verifier 兼容性。生成的状态在此阶段仍是候选。
6. **实例化执行**：为每个候选创建隔离环境、工具集、任务目标和独立 verifier。
   使用概念图和执行图的兼容组合方法生成候选 `action sequence`，再由 trace runner
   在 sandbox 中逐步执行真实工具；不使用真实模型作为 teacher，也不使用任何自报
   完成判断。
7. **环境可解性过滤**：在固定候选生成器版本、harness、预算、任务规范、verifier
   和随机种子策略下进行固定的 100 次独立程序化 rollout。任务环境只有在至少一次
   rollout 通过 verifier，即观测到 `pass@100 > 0` 时，才进入可训练 RL 任务池。
   “独立”必须对应不同的候选序列、参数、分支或随机种子；不能把同一条确定性
   轨迹复制 100 次冒充 `pass@100`。单条固定轨迹使用 `replay_pass` 记录即可。
8. **轨迹质量过滤**：对候选轨迹重新执行或 replay，检查协议、call/result 关联、
   raw action 是否被 parser 修复、状态变化、harness 健康和最终 verifier。只有
   任务成功、关联完整且 harness 健康的轨迹进入普通成功 SFT。
9. **分类保存**：通过样本回写执行图；健康 harness 下的失败样本进入困难度、
   恢复或 RL 分析；harness fault/unknown 样本进入修复和回归集，不给模型错误
   的正负标签。
10. **覆盖反馈**：更新各任务族、抽象状态模式、工具组合、恢复方式、verifier
    模式的覆盖率和成功率，下一轮优先探索未覆盖但程序化 oracle 可解的区域。

`pass@100 > 0` 是任务环境的最低可解性门槛，不是“任务简单”、不是 100 次都
成功，也不是单条 SFT 轨迹自动合格。正向 SFT 仍要求该具体轨迹最终通过
verifier；如果程序化候选生成器只有极低成功率，应标为 `hard_solvable`，优先进入 RL
或诊断集，而不是大量复制为 cold-start SFT。

### Outcome 与训练边界

每个 episode 至少记录：

```json
{
  "task_id": "...",
  "environment_id": "...",
  "initial_state_hash": "...",
  "candidate_generator_version": "...",
  "execution_mode": "live_sandbox",
  "seed": 0,
  "harness_version": "...",
  "tool_schema_version": "...",
  "verifier_version": "...",
  "protocol_status": "valid",
  "call_result_linkage_complete": true,
  "trace_fidelity": true,
  "task_success": true,
  "independent_verifier_passed": true,
  "harness_status": "healthy",
  "failure_class": null,
  "pass_at_100": true,
  "pass_count_100": 1,
  "trajectory": "immutable_event_log"
}
```

普通成功型 SFT 的收录条件为：

```text
task_success
&& protocol_valid
&& call_result_linkage_complete
&& trace_fidelity
&& harness_status == healthy
&& independent_verifier_passed
```

中间工具失败但最终恢复成功的轨迹可以收录，并单独标记
`had_recoverable_tool_failure=true`。最终失败的完整轨迹不作为成功 gold；如果
要训练“正确报告不可用工具”或“从错误中恢复”，必须建立单独的 failure-handling
目标和 verifier，不能混入成功轨迹。

RL 任务环境必须有独立 verifier，且满足 `pass@100 > 0`。后续 student 在健康
harness 上的失败可以作为 RL 的结果信号；harness 故障、verifier 不可用、消息
丢失和无法归因的样本必须重试、replay 或 censored，不能给模型负奖励。

### SFT 与 RL 的职责

- QA/指令数据保持通用语言、知识和约束遵循能力；
- 格式和 tool-call 微样本建立协议冷启动；
- 成功 agent 轨迹教会模型观察—行动循环、状态跟踪、恢复、验证和终止；
- RL 在已知可解的任务环境中优化长程规划、探索、工具选择、效率和结果；
- 失败轨迹默认用于诊断、难度和恢复分析，不因格式正确就当作正向 SFT。

Agent 轨迹同时服务 SFT 和 RL：SFT 提供 cold-start，RL 使用在线 rollout 和
verifier 继续扩大行为分布。具体配比按可训练 assistant token 做小规模 ablation，
不能由概念图大小或 episode 数量直接推断。

### 明确禁止

- 不把格式正确、长度很长或工具很多等同于任务成功；
- 不把模型或程序的自报完成判断当 gold 标签；
- 不把 `pass@100 > 0` 误解为每条 rollout 都成功；
- 不在没有独立 verifier 的情况下把组合出的抽象状态当作事实或 reward；
- 不跨 episode 拼接原始 assistant/tool 文本、shell 输出、session ID 或 call ID；
- 不把 harness fault、工具适配器故障或 verifier 故障作为模型的负样本；
- 不为了图谱向模型协议新增隐藏字段；图谱和 provenance 只在离线数据管线使用。

### 参考

- Kimi K2，§3.1.1：有状态 simulator、rubric 过滤和真实执行 sandbox；
- Kimi K3，§4.2.2：概念知识图谱用于材料检索和任务合成；
- DeepSeek-V3.2，§3.2.3：environment/tool/task/verifier 合成、难度递增和
  `pass@100 > 0` 任务环境过滤。
