1. 原始数据保存在/data/post_train/data/raw/synthetic 
2. 经筛选后可通过的轨迹存在 data/post_train/data/rendered/sft 写为parquet 
3. 过程中如果必须要jsonl格式 可以放在data/post_train/data/jsonl/synth 若需将原始数据转为jsonl
放在data/post_train/data/jsonl/natural 若非必须这么做 则忽略。
4. 已有的原始轨迹数据在/data/rendered/sft/codex_*.parquet。过程中不要删除数据。
5. 相关文件写在scripts/synthesis  
6. 计算节点ssh air-node-03 环境/home/JJ_Group/lih2511/.conda/envs/nanoagent 若有需补齐的环境 请追加在requirements.txt
7. 若生成完毕 将生成parquet合并进model/language_model/data/post_train/data/rendered/sft/ 训练数据是全部model/language_model/data/post_train/data/rendered/sft/*.parquet
8. sft走verl风格 请无视/model中相关实现
