# Harness-Conditioned Agent Post-Training

## 结论

这个方向可行，而且当前仓库已经出现了它最有价值的实验信号：同一个正确动作在 harness 的不同边界上会得到完全不同的结果。但“接入一个更强基模，若它也失败就归因于 harness”只能作为诊断起点，不能单独构成因果归因。

正确的对象不是一个标量 reward，而是一个由模型、harness、环境和 verifier 共同产生的观测：

```text
y = model(prompt)
trajectory = harness(y, environment)
outcome = verifier(trajectory, task)
```

因此需要把 `task success`、`protocol validity`、`tool execution` 和 `harness health` 分开记录。harness 故障的样本不能作为模型的负梯度；无法判断责任时应标为 `unknown/censored`，重放或丢弃，而不是强行给 0 分。

## 当前仓库的事实

- `agent/runtime/src/reward.ts` 目前只做通用完成度检查，没有任务 postcondition。
- `model/language_model/scripts/verl_reward.py` 现在按 task outcome/replay 优先；没有任务契约的历史行会显式返回 `unscored`，不会再获得 JSON 外观奖励。
- `prepare_verl_data.py` 中的 `ground_truth` 是 verl 传给 reward 函数的参考答案字段，不是模型标签。历史 Codex 行没有 workspace snapshot 和独立 verifier，因此现在放入 `unscored_codex_replay` 契约，而不是空字符串或伪造答案。
- 当前 verl launcher 已切到 `tool_agent`/Agent Loop：verl 负责多轮生成、工具响应 token 和 `response_mask`，本项目只提供 workspace 工具与 verifier 适配层。GPU 上的完整 GRPO 仍需用兼容的 Hugging Face/vLLM checkpoint 做 smoke。
- Codex 数据混合了不同机器、不同工具集和大量上下文续写。它适合做格式/SFT 输入，不能直接当作有明确外部成功条件的 held-out agent benchmark。

历史 Codex 行现在会明确标成 `unscored_codex_replay`；`verl_grpo.sh` 默认拒绝没有可执行 verifier 的 train/val 文件。这样错误的 `ground_truth` 不会悄悄变成训练信号。

### `ground_truth` 应该是什么

数学题里它通常是标准答案，代码题里它可以是测试规格或 verifier 输入。对 coding agent，推荐的形式是：

```json
{
  "kind": "environment",
  "task_id": "repo-task-001",
  "workspace": "snapshot-or-image-id",
  "verifier": {"kind": "command", "command": "pytest -q"}
}
```

它描述“如何判定任务完成”，不要求模型复制某个 gold assistant 文本。只有 verifier 在 agent loop 中执行后，`task_outcome` 才能进入 reward；单纯把历史 assistant 回复塞进 `ground_truth` 会奖励模仿轨迹，而不是解决问题。

## Runtime 选型

不从头写训练引擎，也不把 DSH 或 PRIME 当成同一种东西：

| 组件 | 负责什么 | 本项目用法 |
| --- | --- | --- |
| `verl` Agent Loop | 批量 rollout、token mask、log-prob、GRPO/PPO | 训练主引擎，复用 `tool_agent`、`BaseTool`/function tools |
| 本项目薄适配层 | Codex/Qwen/DeepSeek 协议转换、per-episode workspace、任务 verifier | 自己写，保持小而稳定；当前实现见 `verl_workspace_tool.py` |
| DSH | 面向开发者的插件化交互式/Headless harness | teacher 调试、harness patch 探索；不放进 RL 内循环 |
| PRIME | implicit process reward + RL 算法 | outcome verifier 稳定后作为 process-reward ablation，不替代 runtime |

DSH 的核心优势是插件化、持久 session 和交互式开发体验；它不是 verl 高并发 rollout、每条轨迹独立 workspace 和 token-preserving 训练接口的直接替代品。PRIME 是 RL/reward 方法，不是 workspace 执行框架。当前最短路径是复用 verl 已经 vendored 的 `tool_agent` 和 tool parser，只维护一个规范化的 `ToolCall/ToolResult` 层，让 DeepSeek 的 OpenAI calls 与 Qwen 的 Hermes/Qwen calls 进入同一执行内核。

### Codex SFT 与 Qwen rollout 的边界

Codex 是数据来源和行为语义，不是 Qwen 的最终 wire format。应保留一份 canonical event：

```text
user -> assistant ToolCall(name, arguments) -> ToolResult -> ... -> final
```

然后按消费方渲染：NanoGPT 当前的 SFT 编码器可把 tool call 序列化成 JSON 文本；Qwen student 的 SFT/rollout 应使用其 tokenizer 的 chat template 和 Hermes/Qwen tool parser；DeepSeek teacher 通过 OpenAI function-call adapter 进入同一 canonical ABI。不能把 Codex JSON 字符串直接当作 Qwen 原生 tool token 的替代品，否则会把 SFT 与 serving parser 错配，产生“看起来学会 JSON、实际不会调用工具”的假象。

这里有一个当前仓库的硬边界：`prepare_post_train_sft.py` 使用 NanoGPT 自定义 byte-BPE 和本地 safetensors 格式，它不能把 Codex 数据训练成 Qwen checkpoint。若 student 选 Qwen，应读取同一 canonical events，用 `AutoTokenizer.apply_chat_template(..., tools=...)` 渲染，再交给 verl/Hugging Face SFT。今晚不把这一步伪装成已经完成；先用原生 Qwen checkpoint 做 tool-agent smoke，SFT 作为下一步独立实验。

## 对强模型方案的判断

强模型有三个正确用途：

1. 作为 `teacher ceiling`，检查当前工具 ABI、prompt 和环境是否足以支持任务。
2. 生成高质量、覆盖边界情况的轨迹，帮助发现 parser、tool adapter、resume、timeout 和 verifier 的缺陷。
3. 提出 harness patch，但 patch 必须在独立测试和 held-out 任务上验收。

它不能直接证明失败来自 harness，因为强模型也可能遇到任务歧义、工具 schema 不匹配、API 行为变化、环境权限或自身错误。更强模型只能给出 harness 能力的一个上界/下界信号；真正的归因需要同一 raw output 的 replay 或模型×harness 交叉实验。

## 归因实验设计

固定 task、workspace snapshot、tool schema、timeout、temperature 和 verifier 版本，构造：

```text
models  = {scripted/oracle, strong teacher, local student}
harness = {candidate H, reference/replay H, ablation H}
```

对每个 task 运行完整矩阵，并报告：

| 指标 | 含义 |
| --- | --- |
| `task_success` | 外部文件、测试或状态 postcondition 是否满足 |
| `protocol_valid` | raw assistant action 是否被完整解析、call/result 是否配对 |
| `tool_success_rate` | 工具真实返回成功的比例 |
| `recovery_rate` | 工具失败后是否能在预算内恢复 |
| `verifier_score` | 独立 verifier 的任务分数 |
| `harness_fault_rate` | contract test、内部异常、丢消息、超时等框架故障比例 |
| cost/latency | token、工具调用、wall time |

设 `E(m,h)` 为模型 `m` 在 harness `h` 上的 task success。至少要报告：

```text
harness_loss(m) = E(m, H_reference) - E(m, H_candidate)
model_gap(h)    = E(teacher, h) - E(student, h)
```

若想估计交互项，再使用：

```text
interaction = E(T, H_reference) - E(T, H_candidate)
            - E(S, H_reference) + E(S, H_candidate)
```

最干净的 harness 归因是对同一个 raw model response `y` 做两次执行：

```text
verifier(reference_replay(y)) - verifier(candidate_harness(y))
```

这样模型输出被固定，差异才主要落在 parser、工具执行和 runtime。强 teacher 用于提高 `y` 的覆盖率，不能替代 replay。

## Reward 契约

每个 episode 应落盘一个不可变的结构化结果，至少包括：

```json
{
  "task_success": true,
  "protocol_status": "valid",
  "tool_status": "all_succeeded",
  "harness_status": "healthy",
  "failure_class": null,
  "verifier_version": "...",
  "raw_response": "...",
  "trajectory": "..."
}
```

训练使用规则：

- `harness_status=healthy` 且任务失败：这是模型/策略样本，可以进入 RL。
- `harness_status=fault`：重试、reference replay，或把样本权重设为 0；不能给模型负奖励。
- `harness_status=unknown`：暂不用于 advantage，单独进入诊断集。
- `protocol_invalid` 只有在 parser 已经证明是 total、没有丢失正确动作时，才可作为模型的格式能力惩罚。
- JSON 外观、长度和“有最终回答”只能是很小的辅助 shaping，不能替代 task reward。

当前 V1 实现中，明确的 `fault/unknown/censored` 由
`RetryOnIneligibleRewardManager` 抛为可重试的 group failure，并由 replay
buffer refill；模型没有调用 `verify_task` 则记为 `protocol` failure，分数为 0
且仍可用于优化。若关闭 `VERL_RETRY_CENSORED`，只能用于诊断，不能声称训练已隔离 harness 故障。

尤其要保留 raw token/action。若 parser 为了执行而修复多余括号、markdown 或字段名，修复后的 action 不能被当成模型原始 token 直接训练；verl 的多轮 Agent Loop 需要同时保留生成 token、工具响应 token 和 response mask，否则会产生 training-serving mismatch。

## Harness Editing 外循环

把 harness 编辑定义成外层优化问题，而不是让 agent 在训练中随意修改 verifier：

```text
1. 固定 evaluator、任务后置条件和 held-out split。
2. 用 teacher、student 和 scripted oracle 收集失败轨迹。
3. 独立 critic 根据 trace 分类为 model / protocol / tool / environment / harness。
4. 强模型提出最小 harness patch。
5. 在隔离 worktree 中运行 contract、安全、golden trajectory 和 mutation tests。
6. 同时在 train split 与 held-out split 上跑模型×harness矩阵。
7. 只有 task success 提升且 verifier 不变、无安全/回归问题时才接受 patch。
8. 冻结新 harness，再生成 RL trajectory；周期性重复，但不同时漂移模型和 evaluator。
```

验收至少要求：scripted oracle contract 100%；teacher 在 held-out 上有稳定提升；student 提升不能只出现在 teacher 用过的任务；工具失败、超时和 verifier 版本必须可追溯。评测器本身必须在 agent 可编辑 workspace 之外，或由独立进程/签名版本提供。

## 今晚已执行的最小实验

新增 `agent/package/src/diagnose.ts`，固定四个临时 workspace 任务：精确写文件、修复测试、并行创建两个文件、近似 JSON tool call。每个任务都用外部 postcondition 验证，并同时记录 generic verifier、协议、工具和任务指标。

先用 scripted oracle 校准，发现并修复了两个真实 harness 问题：

1. `parseAction` 能恢复文本中的 tool call，但 runner 之前只执行 API 原生 `response.tool_calls`，导致正确的文本动作被追加后反复重试；现在两种入口走同一执行分支。
2. DeepSeek 产生的标准 unified diff 没有 `diff --git` 头，旧实现拒绝它；而且旧的 `execFile(..., {input})` 没有可靠关闭 `git apply` 的 stdin。现在接受标准 diff，并用 `spawn` 明确写入 stdin。

修复前的 `near_json` 校准是 `task_success=0`、`protocol_valid=0`；修复后 scripted 四项为 `4/4`。最新一次 DeepSeek teacher calibration 的四个外部 postcondition 也全部通过、协议全部有效，但其中 `repair_test` 先调用了环境中不存在的 `python`（exit 127），随后用 `python3` 恢复：

```text
task_success       = 4/4
protocol_valid     = 4/4
repair_test        = generic verifier 0.4, external task 1.0
```

这不是能力提升结论，而是一个可复现的 harness calibration 和 bug-fix 结果；它还展示了为什么最终 reward 必须由独立 postcondition 决定，而不能由某次工具失败或通用格式 verifier 决定。原始轨迹在 `logs/harness_diagnose_*.json`，回归检查是：

```bash
cd agent/package
npm run check
npm run diagnose
DIAGNOSE_OFFSET=1 DIAGNOSE_LIMIT=1 npm run diagnose -- teacher
```

另外已加入一个最小的 verl task manifest 和 adapter：

```bash
TASK_MANIFEST=agent/tasks/harness_smoke.jsonl \
  ./model/language_model/scripts/prepare_verl_data.sh
TASK_MANIFEST=agent/tasks/harness_smoke.jsonl \
  TRAIN_BATCH_SIZE=1 TOTAL_TRAINING_STEPS=1 \
  MODEL_PATH=<hf-qwen-checkpoint> \
  ./model/language_model/scripts/verl_grpo.sh
```

它包含四个带独立 shell postcondition 的任务。当前 CPU 工作区只完成了 tool registry、workspace verifier、reward 分支和 launcher guard 的 smoke；没有在本机伪造 GPU GRPO 结果。该 workspace adapter 目前只是受控校准用的 workspace-scoped subprocess，正式大规模运行前仍需 OS/container sandbox。

## 进入 RL 的 gate

在以下条件满足前不启动长训练：

1. 将 `agent/tasks/harness_smoke.jsonl` 扩展到 32 至 128 个明确任务的 train/held-out split，每个任务有独立 postcondition，不能只用 Codex 用户文本。
2. 把 task verifier 和 harness health 写入统一 episode schema；修复样本、超时样本和未知样本不进入普通 advantage。
3. 在 GPU 节点完成 verl 多轮 `tool_agent` 的 1 至 4 step protocol smoke，确认 Qwen tokenizer/chat template、tool parser、workspace 和 response mask 对齐。
4. 完成 `scripted × candidate/reference`、`teacher × candidate/reference`、`student × candidate/reference` 的小矩阵。
5. 先做 1 至 4 step protocol smoke，再做固定预算的短 RL；前后都在同一 held-out 集上报告 task success、recovery 和成本。

## 论文叙事边界

有潜力的叙事是 **Harness-Conditioned Post-Training** 或 **Joint Optimization of Agent Policy and Runtime Interface**：runtime 不是被动容器，而是影响可观测学习信号的可测试组件；强模型用于发现 runtime ceiling，harness patch 改善轨迹分布，冻结后的轨迹再用于 policy optimization。

“模型学会修改自己的 runtime harness”可以作为长期方向，但要把外部 scaffolding 增益和模型内化严格区分：如果换成 reference/minimal harness、换工具 schema 或减少 runtime 帮助后能力不保留，只能说 harness amplification，不能说能力已经内化进 foundation model。当前证据支持先做这个方向的 feasibility study，不支持顶会能力提升 claim。
