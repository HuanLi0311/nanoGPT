# NanoAgent 实验记录

更新时间：2026-08-21

## 历史目标

根据 `NOTICE.md`、`paper/draft.md`、实验日志和提交记录，前一阶段的目标是：

1. 阅读项目约束，建立可复现的 Codex-compatible agent harness。
2. 保留 Codex JSONL 中的消息、tool call、tool result 和 call id，提供受 workspace 边界保护的 `exec_command` 与 `apply_patch`。
3. 接入 SFT 和 verl 风格 GRPO，记录 loss、reward、验证结果和失败原因。
4. 使用本地模型完成训练链路；使用 DeepSeek API 作为 teacher/强 baseline，用相同 prompt、工具、超时和 verifier 做对照。
5. 只有在 held-out 任务的真实工具执行、task success、tool-call validity、recovery rate 和 verifier score 都有证据时，才声称能力提升。

这个目标没有被全部完成。当前已有的是 harness、数据契约、SFT 结果和 verl 协议 smoke；还没有完成可靠的 agent capability gain 结论。

## 已完成的工程结果

- Harness 支持 Codex JSONL 提取、OpenAI-compatible tool calls/results、workspace 边界检查、Codex/unified `apply_patch`、原子 resume state、run id 和 prompt mismatch 拒绝、retry/max-turn 控制、tool-result 保留以及 verifier/reward hook。
- 解析边界能够恢复少量 near-JSON 工具调用，例如多余一个 `}`；这只是输入容错，不等于命令安全或任务成功。
- DeepSeek adapter 已实现并通过本地 HTTP stub 验证请求契约。真实 API episode 后来成功完成一次：`status=done`、一次尝试、仓库检查工具链完成、verifier `score=1`。这只是 harness 冒烟，不是 teacher 对比或训练结果。
- `npm run check` 当前通过：`harness self-check passed`。

### API 驱动的 harness 修复

在 `air-node-02` 通过 `source ~/.bashrc` 调用 DeepSeek API，要求模型在同一轮发出两个独立的 `exec_command`。API 返回 HTTP `200`，且确实返回了两个 tool calls、没有 assistant 文本。原实现会把它们记录为交错的 assistant/tool 消息；这不符合 OpenAI-compatible tool protocol，也会破坏下一轮上下文。

修复后，连续的 assistant tool-call events 会合并为一个带多个 `tool_calls` 的 assistant message，随后按 call id 发送 tool results。runner 也先记录全部 calls，再执行全部 tools；self-check 已覆盖两个并行调用和两个结果。探针记录：`logs/deepseek_parallel_api_probe_20260821.json`。

## 数据结果

清洗后的 Codex 数据由 4,756 个 episode 减为 4,594 个，移除了 unresolved call、orphan result 和无效事件：

| 指标 | 结果 |
| --- | ---: |
| episodes | 4,594 |
| tool calls | 39,785 |
| tool results | 39,785 |
| SFT train tokens | 67,460,899 |
| SFT validation tokens | 3,404,443 |

这次变化是训练输入和数据契约修正，不是能力提升证据。记录：`logs/data_quality_iteration.json`。

## SFT 与本地 GRPO

### 关键修复

SFT 曾把位置 `t` 的 logits 与同一位置的 label 比较，导致 masked loss 虚假偏低而不能正确训练 next-token generation。修复后使用 shifted target：logits at `t` 预测 token `t+1`。

### 结果

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| AdamW pilot, 10 steps | loss `9.889987 -> 1.361520`；fixed masked loss `0.718150` | pipeline/optimizer 检查，不是能力 benchmark |
| Muon pilot, 10 steps | loss `9.962697 -> 14.004341` 或 `14.194479` | 该配置不稳定 |
| aligned SFT, 5 GPUs, 1,000 steps | loss `3.745804 -> 1.020046` | 生成从空白恢复为非空，但没有稳定 JSON tool action |
| aligned checkpoint 一步 GRPO | reward `0.25` | malformed natural-language completion |
| tool-call-only SFT, 5 GPUs, 1,000 steps | loss `0.482588 -> 0.603735`；strict JSON `0/3`；repair parser `3/3` | 只证明格式修复，未执行真实工具 |
| 早期 GRPO smoke | reward `0`；响应为空白或无 JSON action | 没有能力提升证据 |

完整记录包括：`logs/sft_pilot_comparison.json`、`logs/sft_alignment_iteration.json`、`logs/sft_toolcalls_iteration.json`、`logs/rl_smoke_summary.json`。

## verl / Qwen 结果

NanoGPT 的 `best.safetensors` 是自定义 Transformer，不能直接作为 Hugging Face/vLLM verl actor。因此 verl 协议验证使用缓存的 `Qwen/Qwen2.5-1.5B-Instruct` 作为可加载的本地 actor/control；Qwen 本身是预训练/指令微调 checkpoint。NanoGPT 自己的预训练、SFT、GRPO 仍属于另一条本地模型实验线。

### 已成功的单步 smoke

隔离环境使用 Torch `2.10.0+cu128`、vLLM `0.18.1`、verl `0.8.0.dev0`，没有修改节点 base environment。Qwen 单 GPU、rollout `n=2`、一行 train/一行 validation，完成了：

- Ray 和 TransferQueue 初始化
- RLHFDataset filtering/dataloader
- FSDP actor/reference 初始化
- vLLM engine/CUDA graph
- rollout generation
- custom reward、old log probability、advantage
- actor update 和 weight update
- validation

指标：training reward `0.70`，validation reward `0.65`，validation accuracy `0.65`，actor loss `0.0`，actor grad norm `18.295`，吞吐 `6.313 tokens/s`。

限制：只有一条 train/validation 数据、只有一步、reward 是启发式格式 reward、没有保存 checkpoint、没有在 workspace 执行生成命令。因此这只能证明 verl rollout/reward/update 协议通路，不能证明能力提升。记录：`logs/verl_vllm0181_smoke_air-node-04.json`。

### 已知失败边界

- air-node-03 的早期 verl smoke 在 Ray worker/node health 阶段失败；原有 8-GPU NanoGPT pretraining 未被终止，记录：`logs/verl_smoke_air-node-03.json`。
- air-node-04 的多步 fixture 曾因 Ray worker 启动环境的 `init_fs_encoding: unknown encoding: UTF-8` 卡在 Ray 初始化，随后已停止本次实验进程；没有产生多步 reward/loss 结果，不能写成完成。
- 后续 16 train / 4 validation、预期 4-step 复现排除了三个环境问题：显式 UTF-8 环境、离线完整 Qwen snapshot、以及 vLLM 0.18.1 要求 `max_num_batched_tokens >= max_num_seqs`。但 verl V1 `TaskRunnerV1` 仍卡在 actor/TransferQueue 初始化；GPU 未进入 model load，零 rollout/reward/update。使用 `train_batch_size=1` / `ppo_mini_batch_size=1` 重试同样停在此处，因此不是 batch-size 根因。记录：`logs/verl_vllm0181_multistep_air-node-04_iteration.json`。
- air-node-02/03 可以作为后续候选节点，但每次必须先核对具体 GPU 和进程归属，不能终止已有训练。

## DeepSeek API 的正确角色

- Qwen：本地 verl/vLLM actor，提供本地 rollout、log probability、反向传播和权重同步。
- DeepSeek API：teacher/强 baseline，用 harness 执行相同任务，检查 runtime、verifier 和轨迹质量，并用于后续对照。
- API key 只从 `DEEPSEEK_API_KEY` 环境变量读取，不写进仓库、Markdown、日志或 Git 历史。

目前只有一次真实 API harness smoke，没有完成同一 held-out benchmark 上的 DeepSeek 与本地 policy 对照，也没有用 API 结果宣称能力提升。

## 下一步

1. 先固定 held-out Codex task split 和 verifier，明确 workspace task success 定义。
2. 用 Qwen 在具有已验证 V1 worker 初始化的环境复现多步 verl，记录每一步 reward、actor loss、validation 和 checkpoint；不碰 air-node-03 现有训练。air-node-02 的 base vLLM 0.12 不满足当前 verl 要求，air-node-04 当前 V1 worker 初始化仍需修复。
3. 对同一批任务运行本地 policy 与 DeepSeek teacher，统一 prompt、tool、timeout、workspace 和 verifier。
4. 把真实工具执行结果写入 `logs/`，必要时绘制到 `assets/`；在此之前只报告 pipeline、格式和协议结果，不报告 capability gain。

当前修复证明了一个真实 API 驱动的 harness 协议缺陷已被定位并修复，但还没有证明后续 RL 的 trajectory 或 task success 变好。下一次 RL 必须使用修复前后相同的 held-out prompts、同一 reward/verifier，并记录并行调用保留率、工具成功率和最终 task success。
