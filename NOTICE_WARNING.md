# NanoAgent 后训练踩坑与修复记录

更新时间：2026-08-29

本文记录本项目在 Codex 轨迹 SFT、Verl 多轮 GRPO、工具协议和 verifier 上已经实际遇到的问题。目的不是保留调试流水账，而是防止后续实验再次出现“训练能跑，但模型学到的协议与推理协议不同”或“模型完成了任务，reward 仍然全为 0”。

## 先记住结论

Agent RL 的 reward 不是只由模型决定，而是整条链路共同产生：

```text
SFT 数据
  -> tokenizer/chat template
  -> 模型原始输出
  -> tool parser
  -> tool schema/adapter
  -> episode workspace
  -> 独立 verifier
  -> reward manager
  -> GRPO advantage/gradient
```

任何一层错配，都可能把正确动作变成 0 分。看到 `reward=0` 时，不能直接得出“模型能力差”或“GRPO 没梯度”的结论，必须沿这条链逐层定位。

## 已发生问题总表

| 层级 | 犯过的错 | 表现 | 修复 |
| --- | --- | --- | --- |
| SFT 数据 | Parquet 只有 `messages`，没有 Qwen tool template 使用的顶层 `tools`/`enable_thinking` | 能启动 SFT，但训练时没有看到与 GRPO 相同的工具定义 | 增加 schema/template 预检；新 GRPO 先使用官方 Qwen3 后训练模型，旧 4B checkpoint 不再被视为已正确学会当前工具 ABI |
| SFT 模板 | Verl 按 turn 单独套模板再拼接，serving 按完整 conversation 套 Qwen3 模板 | 空 `<think>`、连续 tool result 的边界 token 不一致 | `preflight_sft_template.py` 显式识别差异；不再把 `ignore_input_ids_mismatch=True` 当成真正修复 |
| 模型/Checkpoint | 把 NanoGPT 自定义 tokenizer、`.bin` 或 `safetensors` 与 Hugging Face Qwen/vLLM 混为一条链 | checkpoint 不能加载，或 token/template ABI 不一致 | Qwen SFT/GRPO 只使用同一 HF checkpoint 的 tokenizer 和 chat template；NanoGPT 本地训练保持独立 |
| 工具协议 | Codex 轨迹使用 `cmd/workdir/timeout_ms`，旧 Verl schema 要求 `command/cwd/timeout` | 模型调用语义正确，schema/adapter 仍拒绝 | 保留 Hermes 外层；model-facing schema 改为 Codex 内层字段，adapter 同时接受旧别名 |
| Patch 协议 | Codex `*** Begin Patch` 与标准 unified diff 被当成同一种输入 | 合法 patch 被拒绝 | `WorkspaceTool` 分别支持 Codex patch 和 `git apply` unified diff |
| Workspace | 官方 instruct 模型习惯使用绝对路径 `/workspace` | 命令被判定为逃逸 workspace | 将虚拟 `/workspace` 映射到每个 episode 的真实隔离目录，其他绝对路径仍拒绝 |
| Tool parser | `<tool_call>` 中有坏转义、尾部少 closing tag，或 `arguments` 不是合法 JSON | 工具调用无法执行，后续 verifier 必然失败 | 只修复可确定的非法反斜杠和完整尾部 JSON；可识别工具名但参数坏时执行空参数并记模型/tool failure；不猜不完整 JSON |
| Prompt | 21 个不同任务共用 “Complete the workspace task...” | 模型不知道应创建什么文件、满足什么结果 | 每个 task 改成明确、可执行的语义 prompt |
| Verifier | verifier 要求参考轨迹专用的 `candidate-*` 文件名和隐藏 `.ok` marker | 模型已得到正确最终状态仍判 0 | verifier 改为检查任务语义 postcondition，不检查私有 marker |
| Reward | harness fault、unknown 和普通模型失败混成 0 | 错误负梯度，无法归因 | 记录 `eligible/harness_status/failure_class`；fault/unknown 由 reward manager 重试，不进入普通 advantage |
| 验证 | 只看训练 reward，不保存 held-out 逐任务结果 | 无法判断能力是否随 step 变化 | 每 10 step 保存 checkpoint 和 validation JSONL，记录总 pass rate、reward、工具命中率及逐任务成功率 |
| Ray 资源 | 8 个 FSDP actor 的 placement group 与 reward/storage actor 抢 CPU | GPU 空闲但任务一直 pending | 8 卡 TP=8 配置使用 `RAY_NUM_CPUS=32`，agent/reward/storage worker 各 1 |

## 1. SFT 数据与 Qwen 模板没有真正对齐

### 1.1 字段“能读取”不等于字段“满足模板语义”

现存 `model/language_model/data/post_train/data/rendered/sft/*.parquet` 有两类顶层 schema：

```text
公开 SFT shard:  prompt, prompt_id, messages
Codex/synthetic: messages, data_source, trajectory_id, split, metadata
```

所有现存 shard 都没有顶层 `tools` 和 `enable_thinking`。Verl 的 `MultiTurnSFTDataset` 配置则包含：

```text
messages_key=messages
tools_key=tools
enable_thinking_key=enable_thinking
```

缺列时数据集可以退化为 `tools=None` 和默认 thinking 配置，所以作业不一定报 `KeyError`。真正的问题是：Qwen 的 tool chat template 没有收到与 GRPO rollout 相同的工具 schema。数据里虽然有 assistant 的历史 `tool_calls`，模型却没有稳定学到“看到这份工具定义后，按这份参数 schema 调用工具”的条件关系。

因此旧 4B SFT checkpoint 不能仅凭 loss 下降就被认定为学会了当前工具协议。

### 1.2 per-turn 模板与完整对话模板不一致

Verl SFT 数据集会对每个 message 单独调用 chat template，再拼接 token；推理/rollout 通常对完整 conversation 一次性调用 Qwen template。Qwen3 至少有两类已观察到的差异：

- 完整模板可能插入空的 `<think>...</think>` block；
- 相邻的多个 `tool` message 会被完整模板合并在同一个 user/tool 边界中，而 per-turn 模板会重复插入 `<|im_end|>` 和 `<|im_start|>user`。

`model/language_model/scripts/preflight_sft_template.py` 对 138 个 Parquet 文件抽样 1103 行时得到：

```text
完全一致                                      1
已识别的 Qwen3 模板差异                    1102
其中包含空 think block                     1101
其中包含连续 tool message 边界差异             5
未识别差异                                     0
```

这说明差异是可解释的，但绝不是“模板天然一致”。`ignore_input_ids_mismatch=True` 只是不让训练退出，并选择 per-turn 拼接结果；它不是消除 train-serving mismatch 的修复。

### 1.3 已做修改和后续正确做法

已经完成：

- `filter.py` 把 Codex event 规范化为 `messages`，保留 `tool_call_id`、工具名和 arguments；
- `tool_message.py` 在文本投影时保留 call/result ID，不再只保留普通 content；
- `preflight_sft_template.py` 将已知模板差异和未知差异分开，未知差异直接失败；
- `check_sft_environment.py` 在多卡启动前检查环境、Parquet 和 HF checkpoint；
- Qwen GRPO 使用 HF/Qwen checkpoint，NanoGPT 自定义 tokenizer/checkpoint 不再混入 vLLM/Verl 链路；
- 2026-08-29 的修复验证跳过旧 SFT，直接用官方 Qwen3-1.7B 后训练模型确认 reward 链路。

仍需遵守：

- 若重新做 Qwen SFT，应为每条样本提供与部署一致的顶层 `tools`，并用同一个 `AutoTokenizer.apply_chat_template(..., tools=...)` 渲染；
- SFT preflight 必须在长训练前运行，不能只检查 `messages` 列存在；
- 不允许把旧 NanoGPT `.bin`、本地 byte-BPE 编码结果或自定义 `safetensors` 直接交给 Qwen/Verl；
- 旧 4B checkpoint 已经包含上述历史模板风险，不能用修 adapter 的方式追溯性地修复其 SFT 权重。

## 2. Codex 内层协议与 Hermes 外层协议错配

### 2.1 两层协议不能混为一谈

Qwen/Verl rollout 使用 Hermes 外层：

```text
<tool_call>
{"name":"exec_command","arguments":{...}}
</tool_call>
```

Codex 轨迹主要使用下面的内层参数：

```json
{
  "name": "exec_command",
  "arguments": {
    "cmd": "python3 -m pytest -q",
    "workdir": "/workspace",
    "yield_time_ms": 10000,
    "max_output_tokens": 2000
  }
}
```

旧 `verl_tools.yaml` 却要求：

```json
{"command":"...","cwd":".","timeout":60}
```

所以“模型会调用 `exec_command`”仍然不够；字段名、单位和路径语义只要有一个不一致，工具就不会执行。

### 2.2 修改方式

没有重写 Verl 或另造一套 parser，而是保留 Hermes 外层，只改最薄的 model-facing schema 和 workspace adapter：

- schema 现在向模型要求 Codex 的 `cmd`，并公开 `workdir/yield_time_ms/max_output_tokens/timeout_ms`；
- adapter 同时接受 `cmd` 或 `command`、`workdir` 或 `cwd`、毫秒 `timeout_ms` 或秒 `timeout`；
- `apply_patch` 接受 Codex patch 字符串，也接受标准 unified diff；
- `/workspace` 被解释为当前 episode 的虚拟根目录；`/etc`、其他用户目录等外部绝对路径仍会被拒绝；
- `verify_task` 从模型可见工具列表移除，最终 verifier 改为 out-of-band 自动执行，避免模型自报完成。

相关实现：

```text
model/language_model/config/verl_tools.yaml
model/language_model/scripts/verl_workspace_tool.py
third_party/verl/verl/experimental/agent_loop/tool_parser.py
```

### 2.3 `arguments` 非法 JSON 到底是谁的问题

先看 raw model response：

- raw `<tool_call>` 本身已经是不完整 JSON、引号不闭合或 arguments 类型错误：优先归为模型/SFT/template failure；
- raw JSON 合法，但字段名被 schema 拒绝：adapter/schema failure；
- raw JSON 合法且 tool 执行成功，最终状态正确但 verifier 失败：verifier failure；
- response 被 max token 截断，只有 closing `</tool_call>` 丢失而 JSON 对象完整：parser 可以安全恢复；
- harness 丢消息、call/result ID 错连、内部异常或环境超时：harness fault，不能给模型普通负奖励。

Hermes parser 只做边界明确的恢复：修复 shell 正则中的非法 JSON 反斜杠、读取完整的尾部 JSON 对象。若能找到工具名但 arguments 无法解析，会把它执行为 `{}`，得到一个可评分的模型/tool failure；不会伪造原本不存在的命令。

## 3. Verifier 设计把正确结果判成 0

### 3.1 错误设计

旧 `synthesis-full-v1` verifier 不只检查任务结果，还要求模型复现程序化参考轨迹的私有证据。例如：

- 修好 `buggy.py` 并能 `py_compile`，但没有 `compile-*.ok`，判失败；
- 写出正确 SHA-256，但文件名不是 `checksum-{candidate_index}.txt` 或没有 `checksum-*.ok`，判失败；
- 创建合法 `output.svg`，但没有 `artifact-*.ok`，判失败；
- 正确修复 `calc.add`，但没有运行参考轨迹规定的 marker 命令，判失败。

这些 marker 只能由数据生成器知道，prompt 并未告诉模型。它们验证的是“有没有照抄参考轨迹”，不是“任务是否完成”。

同时，21 个任务共用笼统 prompt：

```text
[domain/subdomain] Complete the workspace task and leave a verifier-checkable result.
```

模型既不知道具体目标，也不知道隐藏 marker。于是工具调用可能成功，reward 仍长期为 0。

### 3.2 修改后的原则

`synthesis-semantic-v2` 遵守以下规则：

1. verifier 只检查用户可见的最终 postcondition；
2. 接受语义等价的合理文件名，除非文件名本身就是任务要求；
3. 代码修复用行为测试或编译测试，不 grep 某一行实现；
4. checksum、排序、CSV 聚合等检查精确内容；
5. verifier 在 agent workspace 外由 harness 自动运行，模型不能修改或调用它；
6. 每个任务必须满足“初始 workspace 失败、参考轨迹通过”；
7. `verifier_version` 随语义变化升级，历史结果不能和新版本静默混算。

具体修改包括：

- 21 个 task 全部改为明确 prompt；
- 去掉 `.ok`、`candidate-*` 等隐藏 marker 依赖；
- syntax repair 改为 `python3 -m py_compile`；
- unit/parser repair 改为直接行为断言；
- checksum 改为与 `sha256sum payload.txt` 的精确 digest 比较；
- artifact/report/manifest/inventory 检查真实内容；
- 在 `test_synthesis.py` 中加入 21 项负例/正例回归。

相关实现：

```text
agent/tasks/synthesis_full.py
agent/tasks/synthesis_full.jsonl
model/language_model/scripts/synthesis/test_synthesis.py
```

## 4. Reward 全零为何会导致梯度全零

GRPO 在同一 prompt group 内比较多个 rollout。若一组 rollout 全部得到相同 reward，标准化后的 advantage 就没有区分度：

```text
同组 reward = [0, 0, 0, 0]
-> advantage 全相同/归零
-> policy gradient 为 0 或接近 0
```

旧 4B 训练对此有直接证据：

```text
v5 validation: 113/113 reward=0
v6 validation:  25/25 reward=0
合计:          138/138 reward=0
v6 step 10-19: critic/score/mean=0.0, actor/grad_norm=0.0
```

把 4B 的旧 workspace 用 semantic-v2 verifier 重放后，`12/964` 能通过。这说明旧 verifier 确实制造了部分假 0；但另外 `952/964` 仍失败，说明问题不能全部甩给 verifier。笼统 prompt、旧 SFT/template 错配和模型没有完成任务也共同存在。

修复后的官方 Qwen3-1.7B 一步探针得到：

```text
训练 reward mean/min/max: 0.375 / 0 / 1
advantage min/max:        -0.866 / +1.500
actor/grad_norm:           3.3005
step 0 eval pass_rate:     3/24 = 0.125
step 1 eval pass_rate:     4/24 = 0.1667
step 1 tool hit rate:      0.5833
eligible/healthy:          24/24, 24/24
```

这证明修复后同组中同时出现成功和失败 rollout，GRPO 获得了正负 advantage 与非零梯度。一步 eval 的变化不能证明能力已经提升，只能证明 reward/gradient 闭环恢复。

证据位置：

```text
logs/grpo_qwen4b_sft_step3600_synthesisfull_codexcompat_eval10_v5/validation/
logs/grpo_qwen4b_sft_step3600_synthesisfull_codexcompat_eval10_v6_retry3/validation/
logs/grpo_qwen3_1_7b_official_semantic_v2_probe_ray32.log
logs/grpo_qwen3_1_7b_official_semantic_v2_probe_ray32/validation/
```

## 5. Harness fault 不能作为模型负样本

应按下面的责任边界处理 episode：

| 状态 | 是否进入普通 GRPO advantage | 说明 |
| --- | --- | --- |
| `harness_status=healthy, task_success=1` | 是 | 正样本 |
| `harness_status=healthy, task_success=0` | 是 | 模型/策略失败样本 |
| `harness_status=fault` | 否 | 重试或 reference replay |
| `harness_status=unknown/censored` | 否 | 诊断后再决定 |
| raw protocol invalid 且 parser 已证明没有丢正确动作 | 可以 | 模型格式能力失败 |

当前实现由 `RetryOnIneligibleRewardManager` 隔离 `eligible=false` 的样本，并记录：

```text
task_success
pass_rate
reward_mean
protocol_status
tool_calls / tool_successes / tool_call_hit_rate
harness_status
failure_class
verifier_version
```

不要只看一个 `reward` 标量。至少要同时检查 raw output、解析出的 tool call、tool result、最终 workspace 和 verifier reason。

## 6. 资源问题也曾伪装成训练问题

以下现象不属于模型能力：

- Ray worker 并发 import Python stdlib/torch 时出现 partially initialized module；通过 `sitecustomize.py` 和 worker preload 预热相关模块；
- `RAY_NUM_CPUS=16/24` 时，8 个 actor placement bundle 与 storage/reward actor 无法同时满足，表现为 GPU 空闲、任务永久 pending；稳定配置改为 32 CPU；
- TP=8 时 actor、vLLM、reward 和 checkpoint 保存阶段显存/CPU 状态差异很大，低 GPU utilization 不等于训练已死；
- FSDP 每个 checkpoint 约 21 GB，NFS 保存一次可超过 3 分钟；保存期间不能因为日志暂时不动就杀进程；
- `max_response_length` 太小会截断 tool call；parser 只能恢复 JSON 已完整、仅 closing tag 缺失的情况。

因此长训练前先跑同配置 1-step probe。probe 必须完成 rollout、reward、actor update、checkpoint、训练后 validation，并自然退出。

## 7. 长训练前强制检查清单

### SFT

- [ ] Parquet 不只是有 `messages`；tool SFT 还要有与部署一致的 `tools` schema。
- [ ] assistant `tool_calls[].function.arguments` 能被规范化为 JSON object。
- [ ] `tool_call_id` 与 tool result 完整配对。
- [ ] 使用目标 Qwen checkpoint 自己的 tokenizer/chat template。
- [ ] 运行 `preflight_sft_template.py`，未知 token mismatch 必须为 0。
- [ ] 不用 `ignore_input_ids_mismatch=True` 掩盖未知差异。
- [ ] 检查 supervised token/loss mask，不只看总 loss。

### Tool/harness

- [ ] raw Hermes tool call 是完整 JSON object。
- [ ] schema 字段与 SFT 中的内层协议一致。
- [ ] `exec_command`、`apply_patch`、`/workspace` 都通过 self-check。
- [ ] verifier 不在模型可见工具列表里。
- [ ] call/result ID、workspace state delta 和 raw response 均落盘。

### Verifier/reward

- [ ] 初始 workspace 必须失败，reference trajectory 必须通过。
- [ ] verifier 检查语义结果，不依赖隐藏 marker 或唯一参考路径。
- [ ] verifier/version、harness/version 和 tool schema/version 一起记录。
- [ ] harness fault/unknown 不进入普通 advantage。
- [ ] validation 保存 `eval/pass_rate`、`eval/reward_mean`、逐任务 success rate 和 tool hit rate。

### GRPO

- [ ] 先运行 1-step、`rollout.n >= 2` 的端到端 probe。
- [ ] 至少一个训练 group 同时包含不同 reward。
- [ ] 检查 advantage 同时有正负值，`actor/grad_norm` 非零。
- [ ] 确认 checkpoint 和 validation JSONL 都落盘后再启动长训练。
- [ ] “batch 中只有 2/3 个 prompt”只是单步采样量，不等于整个任务池只有 2/3 条；仍需单独核对 train/val manifest 行数和任务覆盖率。

## 8. Reward 为 0 时的最短排障顺序

```text
1. raw response 中有没有 <tool_call>？
   没有 -> 模型/SFT/prompt

2. <tool_call> 内是不是合法且完整的 JSON？
   不是 -> 模型/SFT/template，或明确的 token 截断

3. name 和 arguments 是否符合当前 schema？
   不符合 -> schema/adapter 或旧 SFT ABI

4. tool 是否真实执行，exit_code/tool_status 是什么？
   未执行 -> parser/schema/harness
   执行失败 -> 模型命令或环境；按 failure_class 区分

5. workspace 最终状态是否已经满足任务？
   满足但 verifier=0 -> verifier bug
   不满足 -> 模型/策略失败

6. harness_status 是否 healthy、eligible 是否 true？
   否 -> 重试/诊断，不能作为普通负样本

7. 同组 reward 是否全部相同？
   是 -> GRPO 没有可学习的相对信号，先修前面的链路
```

## 9. 不应再做的事

- 不因训练进程成功启动就认定 SFT/GRPO 数据契约正确；
- 不因 loss 下降就认定模型学会工具调用；
- 不把 Codex JSON 文本直接当成 Qwen 原生 tool template；
- 不要求模型猜 hidden marker、candidate index 或参考轨迹文件名；
- 不把模型自报“完成”当 verifier；
- 不把 harness fault 记为模型 reward 0；
- 不在 reward 全相同时讨论学习率、正负梯度或训练步数；
- 不用一步 eval 波动声称能力随训练提升；
- 不在没有 held-out validation JSONL 的情况下跑 100-step 长训练。

