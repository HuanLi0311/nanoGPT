# Controllable Environment–Trajectory Synthesis

## 目标与可检验结论

本方案不问“初始环境多样性和后期 diagnosis 哪个都重要”，而问一个有预算约束的决策问题：

> 在总合成成本、轨迹数、训练 token 和优化步数相同的条件下，下一单位预算用于扩大初始环境支持域，还是用于根据当前策略的失败进行定向合成，哪个更能提高未见任务上的最弱领域性能？两者是替代还是互补？

核心区分是：

- Domain/Material Graph 控制任务语义的支持域（coverage）；
- Trajectory Graph 控制给定环境内工具如何组成任务（composition）；
- diagnosis 不是第三种数据，而是下一轮 synthesis budget 的自适应分配策略。

如果 diagnosis 可以创建新环境，它就是“训练后、由错误反馈驱动的环境多样化”。因此实验比较的是静态的 ex-ante coverage 与自适应的 ex-post allocation，而不是抽象地争论哪个模块更重要。

## 四阶段流水线

### Stage 1：Domain/Material Graph（K3 风格）

输入是显式层级：

```text
domain -> subdomain -> atomic concept
```

每个 concept 绑定一个来源和一个声明式 task recipe。采样器按 profile 中的目标分布使用 largest-remainder allocation 分配精确配额，再以固定 seed 选择 concept。当前支持：

- `web`：HTTP(S) 页面，记录最终 URL、ETag/Last-Modified、内容 hash 和抓取时间；
- `repo`：本地 repository 文件或 glob，记录 commit（若可取得）；
- `document`：本地文本或 PDF；PDF 使用系统 `pdftotext`；
- `inline`：仅用于固定 fixture、单元测试和完全程序化材料。

输出：

```text
stage1/domain_graph.json
stage1/materials.jsonl
stage1/materials/*.txt
```

`materials.jsonl` 保存 domain、subdomain、concept、material ID、SHA-256、license、resolved URI、revision 和 task recipe。相同来源只抓取一次；不同采样实例保留独立 material ID。

### Stage 2：Task/Environment Construction

task recipe 将材料包确定性地变成：

```text
material package
  -> initial workspace files
  -> task prompt / goal facts
  -> available tools
  -> independent verifier
  -> action patterns
```

模板只替换明确 token，例如 `{{material}}`、`{{material_id}}`、`{{material_sha256}}`、`{{source_uri}}`。原始材料和来源 URL 只允许进入 prompt 或初始文件，不能插入 action/verifier 命令。任务必须声明非空 verifier、workspace 内相对路径和已实现的工具。当前真实 adapter 只有：

```text
exec_command, apply_patch
```

新合成任务默认声明 `sandbox_backend: bwrap`。Stage 3 会先在隔离后的初始状态运行 verifier；初始状态已经通过、verifier 自身故障或路径逃逸的任务一律拒绝。这样 verifier 检查的是 agent 带来的状态变化，而不是天然成立的断言。

输出：

```text
stage2/tasks.jsonl
```

### Stage 3：Trajectory Graph（Agent-World 风格）

每个 action 是一个节点，边来自三类约束：

- `preconditions` / `effects` 的事实依赖；
- `depends_on` 的显式动作依赖；
- `{{output:action_id}}` 的参数依赖。

采样器从 initial facts 出发，只扩展当前可执行节点，直到 target facts 与 required actions 同时满足：

- `goal`：优先选择能缩短目标距离的节点；
- `uniform`：对当前可执行节点均匀随机排序，作为 composition ablation。

参数依赖在真实执行时实例化；插入 `exec_command` 的上游输出会先 shell-quote。候选路径随后使用与 Verl 相同的 `WorkspaceTool` 在 fresh workspace 中执行：Bubblewrap 只挂载只读系统运行库与可写 `/workspace`，清空宿主环境变量并隔离 network/PID 等 namespace；out-of-band verifier 使用同一 sandbox。抽象路径不能充当 gold。

接受条件为：

```text
task_success
and independent_verifier_passed
and harness_status == healthy
and complete call/result linkage
and trace fidelity
```

输出：

```text
stage3/trajectory_graph.json
stage3/oracle_episodes.jsonl       # 仅为可解性证据
stage3/validated_tasks.jsonl
stage3/rejected_tasks.jsonl
```

### Stage 4：Training Data

Programmatic oracle 只证明存在一条可执行解，不能代表目标模型会解决任务，也不能直接进入 SFT。

训练数据分成两条路径：

1. **SFT**：teacher 或 current policy 在真实工具环境中 rollout；只有独立 verifier 通过、harness 健康、call/result 完整的轨迹进入 `sft.jsonl` 和 Parquet。`programmatic_oracle` provenance 会被硬拒绝。
2. **RL**：所有 oracle-validated 环境进入 `rl_tasks.jsonl`，交给现有 Verl tool-agent 生成真正的 on-policy rollouts 和 reward。

输出：

```text
stage4/rl_tasks.jsonl
stage4/teacher-or-current-rollouts.jsonl
stage4/sft.jsonl
stage4/train_sft-four-stage.parquet
stage4/test_sft-four-stage.parquet
stage4/rejected_rollouts.jsonl
stage4/training_data_report.json
```

旧的 `model/language_model/scripts/synthesis/synthesize.py` 会把 programmatic trace 标记为 `programmatic_oracle` SFT；该文件保留作旧实验复现，本 proposal 的训练结论只使用 `scripts/synthesis/runner.py` 的 Stage 4 gate。

## Plan 契约

最小 plan 是一个 JSON 文件：

```json
{
  "profiles": {
    "diverse": {"count": 1, "distribution": {"coding.repair": 1}}
  },
  "domains": [{
    "name": "coding",
    "subdomains": [{
      "name": "repair",
      "concepts": [{
        "name": "parser-regression",
        "source": {"kind": "repo", "uri": "./corpus/repo", "glob": "**/*.py", "license": "MIT"},
        "task": {
          "id": "record-material-hash",
          "prompt": "Compute the SHA-256 of context.txt and save it in digest.txt.",
          "files": {"context.txt": "{{material}}"},
          "available_tools": ["exec_command"],
          "sandbox_backend": "bwrap",
          "verifier": {"command": "test \"$(cat digest.txt)\" = {{material_sha256}}"},
          "initial_facts": ["workspace:ready"],
          "target_facts": ["file:digest.txt:exists"],
          "actions": [{
            "id": "write_digest",
            "tool": "exec_command",
            "arguments": {"cmd": "sha256sum context.txt | awk '{print $1}' > digest.txt"},
            "preconditions": ["file:context.txt:exists"],
            "effects": ["file:digest.txt:exists"]
          }]
        }
      }]
    }]
  }]
}
```

Recipe 的 actions 是可信的程序化 oracle，不是发给最终策略的答案。实际 policy 只看到 prompt、初始 workspace 和 `available_tools`。

## 运行方式

使用仓库已有环境：

```bash
PY=/home/JJ_Group/lih2511/.conda/envs/nanoagent/bin/python
```

构建窄或多样 base，并执行 oracle gate：

```bash
$PY scripts/synthesis/runner.py prepare PLAN.json runs/base-diverse \
  --profile diverse --seed 7 --path-policy goal
```

使用任意 OpenAI-compatible vLLM/API endpoint 运行真实 teacher/current policy。策略只得到 prompt 和该任务的 `available_tools`，不会得到 oracle path：

```bash
$PY scripts/synthesis/policy_rollout.py \
  runs/base-diverse/stage3/validated_tasks.jsonl \
  runs/base-diverse/stage4/teacher-rollouts.jsonl \
  --base-url http://127.0.0.1:8000/v1 --model MODEL_ID --policy-kind teacher
```

policy-rollout JSONL 带有完整 `messages`、tool events、policy provenance 和 out-of-band outcome。Stage 4 也可以导入现有 Verl validation JSONL（`input/output/task_success/harness_status`）。将通过轨迹制作成 SFT：

```bash
$PY scripts/synthesis/runner.py finalize runs/base-diverse \
  runs/base-diverse/stage4/teacher-rollouts.jsonl \
  --policy-kind teacher --model MODEL_ID
```

从 current-policy 失败生成下一轮领域分配。`runs/diagnostic-arena` 必须使用同一 taxonomy，且是所有实验条件共用的、领域覆盖完整并与训练材料隔离的 arena；不能直接用 narrow base 自己的任务诊断，否则它永远看不到 base 缺失的领域：

```bash
$PY scripts/synthesis/runner.py diagnose runs/diagnostic-arena POLICY_ROLLOUTS.jsonl \
  --output runs/diagnostic-weights.json

$PY scripts/synthesis/runner.py prepare PLAN.json runs/base-diverse-diagnosed \
  --profile diverse --weights runs/diagnostic-weights.json --seed 8 --path-policy goal
```

RL 继续使用现有入口；一个 condition 对应一个统一 checkpoint，并在全部 benchmark 上评测：

```bash
TASK_MANIFEST=$PWD/runs/base-diverse/stage4/rl_tasks.jsonl \
  ./model/language_model/scripts/verl_grpo.sh
```

自检：

```bash
$PY scripts/synthesis/test.py
$PY -m model.language_model.scripts.synthesis.test_synthesis
```

## 主实验与必要消融

### Base diversity × diagnosis 的 2×2

所有 cell 使用相同 pretrained checkpoint、总任务数、训练 token、更新步数、平均工具数和最终评测集。`uniform` 对照也必须得到同样数量的第二轮数据，不能用“不继续训练”代替。

| | 第二轮 uniform synthesis | 第二轮 diagnosis-guided synthesis |
|---|---:|---:|
| Narrow base | `N-U` | `N-D` |
| Diverse base | `D-U` | `D-D` |

预注册 contrasts：

```text
diagnosis under narrow = Y_ND - Y_NU
diagnosis under diverse = Y_DD - Y_DU
base effect            = Y_DU - Y_NU
interaction            = (Y_DD - Y_DU) - (Y_ND - Y_NU)
allocation decision    = Y_ND - Y_DU
```

- interaction `< 0`：diagnosis 能替代一部分初始多样性；
- interaction `> 0`：两者互补，广 base 是有效 diagnosis 的前提；
- interaction约为 `0`：收益近似可加，但仍可比较单位成本收益；
- `Y_ND - Y_DU` 直接回答固定预算应前置扩覆盖，还是后置定向修补。

### Stage 1 与 Stage 3 的机制消融

Stage 1 使用 `narrow/diverse` profile；Stage 3 使用 `uniform/goal` path policy。两者交叉形成另一组 2×2，用来区分：

- 性能来自材料和领域语义本身；
- 还是来自相同工具被组合成更多目标一致路径。

每个 cell 必须匹配 task 数、轨迹长度分布和 verifier 通过门槛。额外报告 unique domain/subdomain/concept、有效领域数 `exp(H(p))`、unique material hash、工具组合和路径长度分布，避免把数量误报成语义多样性。

## 评测与防泄漏

主指标不是简单平均分，而是 normalized benchmark score 的 worst-domain 或 bottom-20% CVaR；同时报告 macro average、每个 benchmark 向量和未见环境 OOD。若只有 A/B：

```text
primary = min(score_A, score_B)
constraint: score_A_after - score_A_before >= -epsilon
```

Diagnosis 只能读取所有 condition 共用的、覆盖完整的独立 dev/diagnostic environments，不能查看最终 benchmark test item、实体、workspace 或 verifier。诊断集合不能由各自训练 base 决定，否则 narrow condition 会出现结构性盲区。若使用 benchmark dev split，应明确把结论限定为 benchmark-aware adaptation，不能声称一般能力发现。

## 当前安全边界

本流水线生成的新任务默认使用系统已有的 Bubblewrap：workspace 是唯一可写的持久挂载，`/usr`、`/bin`、`/lib*` 只读，`/tmp` 为临时内存文件系统，宿主环境变量被清空，network/PID 等 namespace 被隔离。Web 材料仍只允许进入 prompt 或初始文件，不能插值进 action/verifier 命令。

为复现既有数据，缺少 `sandbox_backend` 的旧 manifest 仍默认为 `workspace_host`，它不是 OS sandbox，只能运行可信任务。Bubblewrap 路径也没有 cgroup/配额级资源治理；当任务需要额外编译器依赖或面对完全不可信 workload 时，应增加显式只读挂载并在生产容器中设置 CPU、内存和进程上限，现有任务与训练数据契约无需改变。
