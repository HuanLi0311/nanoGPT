1. 原始数据保存在/data/post_train/data/raw/synthetic 
2. 经筛选后可通过的轨迹存在 data/post_train/data/rendered/sft 写为parquet 
3. 过程中如果必须要jsonl格式 可以放在data/post_train/data/jsonl/synth 若需将原始数据转为jsonl
放在data/post_train/data/jsonl/natural 若非必须这么做 则忽略。
4. 已有的原始轨迹数据在/data/rendered/sft/codex_*.parquet。过程中不要删除数据。
5. 相关文件写在scripts/synthesis  
6. 计算节点ssh air-node-03 环境/home/JJ_Group/lih2511/.conda/envs/nanoagent 若有需补齐的环境 请追加在requirements.txt
7. 若生成完毕 将生成parquet合并进model/language_model/data/post_train/data/rendered/sft b并执行sft 训练数据是全部model/language_model/data/post_train/data/rendered/sft/*.parquet
8. sft走verl风格 请无视/model中相关实现

当前 synthesis 的方法边界见 `PROPOSAL.md`：Stage 1 复现 Kimi K3 §4.2.2 的递归知识图谱、可控节点采样与 Web material retrieval；Stage 2–3 复现 Agent-World §3.1/§3.1.1 的环境任务构造和强/弱/独立依赖加权图。每次运行的 `report.json`、`REPORT.md` 和 Stage 1 雷达图记录 verifier 过滤前规模。

## Workspace harness boundary

`agent/workspace` is the file-level workspace boundary. Each episode owns one
fresh root, and relative paths plus `/workspace/...` are resolved below that
root. Absolute paths outside it and symlink components are rejected.

Legacy tasks use `workspace_host`, which is not an OS sandbox. New synthesis
tasks set `sandbox_backend=bwrap`: only the workspace is writable, platform
runtime paths are read-only, the host environment is cleared, and network/PID
namespaces are isolated. Bubblewrap availability is required for those tasks.

The model-facing workspace ABI is defined in
`runtime/src/tools/definitions.ts`: `exec_command` uses `cmd` and `workdir`,
`apply_patch` keeps the Codex patch payload, and `verify_task` is never exposed
to the policy. `command`/`cwd` remain adapter-only compatibility aliases.

The interaction loop is `runtime/src/core_loop.ts`; persistence and retries are
in `runtime/src/runtime.ts`; verifier and reward stay out-of-band. The Python
Verl adapter records the same call/result IDs and workspace state transitions.
