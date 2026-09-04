# Controllable Environment–Trajectory Synthesis

## 目标与可检验结论

本方案不问“初始环境多样性和后期 diagnosis 哪个都重要”，而问一个有预算约束的决策问题：

> 在总合成成本、轨迹数、训练 token 和优化步数相同的条件下，下一单位预算用于扩大初始环境支持域，还是用于根据当前策略的失败进行定向合成，哪个更能提高未见任务上的最弱领域性能？两者是替代还是互补？

核心区分是：

- Domain/Material Graph 控制任务语义的支持域（coverage）；
- Trajectory Graph 控制给定环境内工具如何组成任务（composition）；
- diagnosis 不是第三种数据，而是下一轮 synthesis budget 的自适应分配策略。

如果 diagnosis 可以创建新环境，它就是“训练后、由错误反馈驱动的环境多样化”。因此实验比较的是静态的 ex-ante coverage 与自适应的 ex-post allocation，而不是抽象地争论哪个模块更重要。

## 三段合成与训练门

前三段是数据合成；Stage 4 只是训练数据门，不计入合成规模。方法归属固定为：

```text
Stage 1     Kimi K3 §4.2.2 Knowledge-Graph-Guided Task Synthesis
Stage 2–3  Agent-World §3.1 / §3.1.1 Environment-Task Discovery
Stage 4     本项目已有的 SFT / RL 接口
```

这里的“复现”指公开论文所描述机制的可执行复现，不声称恢复论文未公开的 prompts、内部模型、搜索基础设施或私有数据。

### Stage 1：Kimi K3 Knowledge-Graph-Guided Synthesis

从少量 coarse seed 开始，`expand-graph` 为每个非 atomic 节点搜索 Web，再让外部 synthesis model 输出 finer children、atomic 判断和 related concepts。已有等价节点会被复用；边只允许由 coarse 指向 finer，形成 DAG。递归受 `max_nodes`、`max_depth` 和 `children_per_node` 控制。

物化后的层级为：

```text
domain -> subdomain -> ... -> atomic concept
```

采样时可以单独选择节点，也可以把 related concepts 联合成 keyword set；查询同时包含当前 concept、相关节点和祖先上下文。采样器按 profile 中的目标分布使用 largest-remainder allocation 分配精确配额，再以固定 seed 选择节点和材料。当前支持：

- `web_search`：不需要页面 URL；默认用 `domain + subdomain + concept` 生成查询，搜索 top-K 候选，按 allow/block domain 过滤并去重，不同采样实例优先选择不同页面；
- `web`：已知 HTTP(S) 页面，适合固定语料和可复现实验；
- `repo`：本地 repository 文件或 glob，记录 commit（若可取得）；
- `document`：本地文本或 PDF；PDF 使用系统 `pdftotext`；
- `inline`：仅用于固定 fixture、单元测试和完全程序化材料。

输出：

```text
stage1/domain_graph.json
stage1/discovery.jsonl
stage1/materials.jsonl
stage1/materials/*.txt
stage1/domain_distribution.png
stage1/domain_distribution.pdf
```

`discovery.jsonl` 保存 query、provider、搜索请求及完整候选 URL/rank；`materials.jsonl` 保存最终选中的 URL、domain、subdomain、concept、material ID、SHA-256、license、revision 和 task recipe。默认要求 URL 与规范化内容都唯一；同一 concept 的重复采样会依次使用不同 query 和 candidate，候选耗尽时失败，不用重复页面填满 20K。

无配置时 `web_search` 先尝试无密钥的 DuckDuckGo HTML endpoint；遇到反自动化页面或网络错误时回退到 Wikipedia 官方搜索 API，并在 provenance 中记录原因。规模化运行建议将 `search_endpoint` 指向自建 SearXNG 或有 SLA 的 JSON 搜索 API，并用 `search_params`、`api_key_env`、`api_key_header` 配置；代码同时解析 SearXNG 风格 `results` 和 Brave 风格 `web.results`。搜索只做 shallow discovery 并抓取被选页面，不递归跟随站内链接。

#### 20K 初始分布

第一轮使用 15 个 coarse domains：软件工程、系统与 DevOps、网络安全、数据与数据库、Web 与信息检索、办公与文档、商业与金融、电商与物流、数学与自然科学、工程与制造、医疗与生命科学、法律与公共管理、教育与人文、媒体与通信、旅行与地理服务。初始 profile 在顶层近似均匀分配 20,000 条，每类 1,333 或 1,334 条；后续 diagnosis 才改变分配。
`distribution` 可以全部使用顶层 `domain` key，也可以全部使用 `domain.subdomain` key，不能混用；省略时默认按顶层 domain 均匀分配，避免 subdomain 较多的领域天然占据更多配额。

| Coarse domain | Seed subdomains |
|---|---|
| Software Engineering | repair, testing, refactoring, build/package, code review |
| Systems & DevOps | shell, process, container, deployment, observability |
| Cybersecurity | access control, configuration audit, log analysis, vulnerability triage |
| Data & Databases | SQL, schema, ETL, cleaning, analytics |
| Web & Retrieval | search, browsing, extraction, evidence comparison |
| Office & Documents | text, PDF, spreadsheet, slides, calendar |
| Business & Finance | accounting, markets, operations, reporting |
| Commerce & Logistics | catalog, order, inventory, shipping, supply chain |
| Math & Natural Science | mathematics, statistics, physics, chemistry, earth science |
| Engineering & Manufacturing | specification, BOM, quality control, maintenance |
| Health & Life Science | biomedicine, public health, clinical administration |
| Law & Public Administration | regulation, contracts, forms, public records |
| Education & Humanities | teaching, history, literature, language |
| Media & Communication | email, publishing, social content, media metadata |
| Travel & Geospatial | maps, itinerary, weather, places, local services |

每条材料还记录正交的 `capability` 与 `interface` 标签。benchmark weakness 映射到 `domain × capability × interface` 后再增配，不能围绕 benchmark 测试题本身生成。

### Stage 2：Agent-World Environment/Task Construction

task recipe 将材料包确定性地变成：

```text
material package
  -> initial workspace files
  -> task prompt / goal facts
  -> available tools
  -> independent verifier
  -> action patterns
```

每份材料可通过 `tasks_per_material` 生成多个 task/environment variants；同一模板可用 `{{variant_index}}` 参数化，也可以显式提供多个 templates。模板只替换明确 token，例如 `{{material}}`、`{{material_id}}`、`{{material_sha256}}`、`{{source_uri}}`。原始材料和来源 URL 只允许进入 prompt 或初始文件，不能插入 action/verifier 命令。任务必须声明非空 verifier、workspace 内相对路径和已实现的工具。当前真实 adapter 只有：

Kimi seed graph 不需要预写 task。`prepare --base-url ... --model ...` 会把每条材料作为不可信数据交给 synthesis model，一次生成指定数量的 environment/task recipes；失败记录在 `stage2/rejected_environment_generation.jsonl`。显式 recipe 路径仍保留，作为不依赖外部模型的可重复对照。

```text
exec_command, apply_patch
```

新合成任务默认声明 `sandbox_backend: bwrap`。Stage 3 会先在隔离后的初始状态运行 verifier；初始状态已经通过、verifier 自身故障或路径逃逸的任务一律拒绝。这样 verifier 检查的是 agent 带来的状态变化，而不是天然成立的断言。

这是 Agent-World 环境构造的 workspace adaptation：Web material 对应环境数据库，recipe 对应数据库约束下的工具/任务定义。论文中的任意 Python tool generation、unit-test cross-validation 和数据库多轮 complexification 尚未伪装成已完成；在增加动态 Verl tool ABI 前，领域多样性不等于工具接口多样性。

输出：

```text
stage1/materials_with_task_templates.jsonl  # 保持与 material files 同一相对路径根
stage2/tasks.jsonl
stage2/rejected_task_construction.jsonl
stage1/rejected_environment_generation.jsonl
```

### Stage 3：Agent-World Weighted Tool Graph

每个可实例化 action pattern 是一个节点。图对不同节点建立有向边，并按 Agent-World 的三类关系赋权：

- 强依赖 `weight=3`：`depends_on` 或 `{{output:action_id}}` 参数严格依赖；
- 弱依赖 `weight=2`：前一节点 effects 与后一节点 preconditions 相交；
- 独立边 `weight=1`：其余节点对，用于保持连通。

采样器从 initial facts 出发，只扩展当前可执行节点，直到 target facts 与 required actions 同时满足：

- `agentworld`：按边权进行随机 walk，并在死路时回溯，保证最终候选可达声明目标；
- `goal`：优先选择能缩短目标距离的节点，作为定向对照；
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
stage3/trajectory_graphs.jsonl
stage3/candidate_trajectories.jsonl  # verifier 前的固定规模快照
stage3/oracle_episodes.jsonl       # 仅为可解性证据
stage3/validated_tasks.jsonl
stage3/rejected_tasks.jsonl
stage3/rejected_trajectories.jsonl
```

`trajectories_per_task` 控制每个任务的候选数。报告同时给出 candidate 总数、完整 path signature 数、结构 signature 数和 tool sequence 数，防止把相同结构的重复执行误报成新的组合多样性。
`min_steps`/`max_steps` 控制 walk 长度；提高 `min_steps`，或增加 weak/independent 可执行节点，构成 Agent-World 式 difficulty scaling。若图本身只有唯一必经路径，多次采样仍会被结构 signature 揭示，不能宣称获得了新的组合。

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

`scripts/synthesis/k3_15domain_seeds.json` 是不带 task recipe 的 Stage-1 seed plan；它可直接用于 `expand-graph` 与 `collect`。下面展示的是带显式 recipe、可以不调用 synthesis model 的完整 plan branch：

```json
{
  "require_unique_materials": true,
  "expansion": {"tasks_per_material": 2, "trajectories_per_task": 3},
  "profiles": {
    "pilot": {"count": 200, "distribution": {"coding.repair": 1}},
    "full": {"count": 20000, "distribution": {"coding.repair": 1}}
  },
  "domains": [{
    "name": "coding",
    "subdomains": [{
      "name": "repair",
      "concepts": [{
        "name": "parser-regression",
        "atomic": false,
        "related_concepts": [],
        "source": {
          "kind": "web_search",
          "max_results": 50,
          "queries": [
            "{{concept}} {{subdomain}} {{domain}} documentation",
            "{{concept}} {{subdomain}} {{domain}} case study"
          ],
          "blocked_domains": ["example-spam.invalid"],
          "license": "unknown"
        },
        "task": {
          "id": "record-material-hash-{{variant_index}}",
          "prompt": "Compute the SHA-256 of context.txt and save it in digest-{{variant_index}}.txt.",
          "files": {"context.txt": "{{material}}"},
          "available_tools": ["exec_command"],
          "sandbox_backend": "bwrap",
          "verifier": {"command": "test \"$(cat digest-{{variant_index}}.txt)\" = {{material_sha256}}"},
          "initial_facts": ["workspace:ready"],
          "target_facts": ["file:digest-{{variant_index}}.txt:exists"],
          "actions": [{
            "id": "write_digest",
            "tool": "exec_command",
            "arguments": {"cmd": "set -- $(sha256sum context.txt) && printf '%s\\n' \"$1\" > digest-{{variant_index}}.txt"},
            "preconditions": ["file:context.txt:exists"],
            "effects": ["file:digest-{{variant_index}}.txt:exists"]
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
$PY scripts/synthesis/runner.py expand-graph \
  scripts/synthesis/k3_15domain_seeds.json PLAN.json \
  --base-url http://127.0.0.1:8000/v1 --model SYNTHESIS_MODEL \
  --max-nodes 2000 --max-depth 4 --searches-per-node 3

# 只复现/审计 Kimi K3 Stage 1 时使用；输出中立即包含雷达图。
$PY scripts/synthesis/runner.py collect PLAN.json runs/k3-pilot-stage1 \
  --profile pilot --seed 7

# 三段连跑。PLAN 无预写 recipes 时，同一模型负责 Agent-World-style
# environment/task recipe synthesis；有 recipes 时省略 --base-url/--model。
$PY scripts/synthesis/runner.py prepare PLAN.json runs/pilot \
  --profile pilot --seed 7 --path-policy agentworld \
  --tasks-per-material 2 --trajectories-per-task 3 \
  --base-url http://127.0.0.1:8000/v1 --model SYNTHESIS_MODEL
```

先运行 `200 → 400 → 1,200` pilot；通过来源集中度、重复率、路径结构数和 verifier yield 检查后，再将 profile 改为 `full`，得到目标 `20,000 → 40,000 → 120,000`。运行目录根部的 `report.json` 与 `REPORT.md` 汇报三阶段 verifier 前数量；Stage 1 雷达图叠加 target 与 realized domain counts。

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
  --profile full --weights runs/diagnostic-weights.json --seed 8 --path-policy agentworld
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

Stage 1 使用 `narrow/diverse` profile；Stage 3 使用 `uniform/agentworld` path policy。两者交叉形成另一组 2×2，用来区分：

- 性能来自材料和领域语义本身；
- 还是来自相同工具被组合成更多目标一致路径。

每个 cell 必须匹配 verifier 前候选数；用于训练时还必须匹配 verifier 后的 task 数、训练 token、轨迹长度分布和通过门槛，不足的 cell 继续 over-generate。额外报告 unique domain/subdomain/concept、有效领域数 `exp(H(p))`、unique material hash、工具组合和路径长度分布，避免把数量误报成语义多样性。

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
