<h1 align="center">NanoAgent</h1>

<p align="center"><strong>从原始语料到后训练，把 Agent 框架搭建与底层基模训练的完整链路收敛为最简形式的文件目录。</strong></p>

<p align="center">
  一个为理解而建的视觉-语言模型实现：不靠巨型训练框架，<br />
  从 Byte-level BPE、token 分片、预训练、SFT 到采样生成：<strong>每一步都能追到代码(< 100行)</strong>
</p>

<p align="center">
  <code>Python 3.10+</code> &nbsp; <code>PyTorch 2.2+</code> &nbsp; <code>Decoder-only Transformer</code> &nbsp; <code>MIT</code>
</p>

<p align="center">
  <a href="#一条完整的主线">全链路</a> ·
  <a href="#为什么要把文件压小">100 行原则</a> ·
  <a href="#语言模型从数据到后训练">语言模型</a> ·
  <a href="#视觉支线同样的可读性原则">视觉支线</a>
</p>

---

`NanoAgent` 的 **nano** 指代码表面积，而不是对模型能力的夸张承诺。它以语言模型为主线，将字节级 BPE、二进制 token 分片、Decoder-only Transformer、分布式预训练、带掩码的 SFT 与 temperature/top-k 采样拆成可单独阅读的组件。读者可以从入口一路追到注意力公式，始终知道数据在哪里变形、梯度在哪里流动、checkpoint 里保存了什么。

## 一条完整的主线

```text
原始语料 / 多轮 messages
          |
          |  Byte-level BPE
          v
uint32 token shards + metadata.json
          |
          |  next-token prediction
          v
Decoder-only Transformer
          |
          |  checkpoint + tokenizer fingerprint
          v
监督微调（只计算 assistant token 的损失）
          |
          |  autoregressive sampling
          v
文本生成
```

这条路径刻意不省略关键步骤：预训练回答“如何续写 token”，SFT 回答“哪些 token 应该被监督”，推理回答“如何从 logits 采样”。每一步都应有自己的输入、输出和可检查的元数据，而不是依赖隐含约定。

## 为什么要把文件压小

这个项目的工程约束是：**一个文件只解释一个概念，目标是每个 Python 源文件不超过 100 行。** 直接降低了理解和模型底层逻辑的成本：

- `rope.py` 只做旋转位置编码；
- `attention.py` 只做 Q/K/V、因果注意力和投影；
- `layers.py` 只组合残差、归一化和 MLP；
- `tokenizer.py` 只负责词表、指纹与 embedding；
- 训练入口只负责把已有部件接成一次运行。


## 语言模型：从数据到后训练

### 1. Tokenizer 与数据契约

Tokenizer 独立放在 `language_model/checkpoints/` 下，编码数据与其 `metadata.json` 则留在各自的 `data/encode/` 目录。`language_model/model/tokenizer.py` 会恢复词表中已有的 chat special tokens，保证训练、GRPO 与生成对 `<|eos|>`、`<|im_start|>`、`<|im_end|>` 使用同一组 ID。

`language_model/scripts/prepare_data.py` 负责把数据变成训练时真正读取的格式：

| 场景 | 输入约定 | 输出约定 |
| --- | --- | --- |
| 预训练 | 递归读取 Parquet；文本列为 `text` 或 `raw_content` | `train_*.bin` / `validation_*.bin`，`uint32` token IDs |
| SFT | `train_sft-*.parquet` 与 `test_sft-*.parquet`，包含 `messages` 列 | `input_*.bin`（`uint32`）和 `labels_*.bin`（`int32`） |

SFT 数据会被格式化为：

```text
<|im_start|>{role}
{content}<|im_end|>
```

只有 assistant 消息覆盖的 token 会保留标签；其他 token 写为 `-100`，由交叉熵的 `ignore_index` 忽略。这是后训练与“把整段对话当普通文本续写”之间最关键的区别。

配置入口在：

```text
language_model/config/pretrain.yaml
language_model/config/sft.yaml
language_model/config/rl.yaml
```

其中定义了原始数据目录、分片大小、上下文长度、模型宽度和深度、batch size、步数、学习率、梯度裁剪与 checkpoint 路径。数据、模型和训练参数各自有明确归属，避免把实验参数散落在脚本正文中。

### 2. Decoder-only Transformer

语言模型的可读核心位于 `language_model/model/`：

```text
token IDs
  -> token embedding
  -> [LayerNorm -> causal self-attention with RoPE -> residual
      LayerNorm -> SiLU MLP                     -> residual] x N
  -> LayerNorm + vocabulary projection
  -> logits
```

实现保留了 Transformer 真正需要理解的部件：

- `decoder/attention.py`：多头 Q/K/V、`scaled_dot_product_attention` 与因果掩码；
- `decoder/rope.py`：对 Q/K 应用 Rotary Position Embedding；
- `decoder/layers.py`：预归一化残差块和 SiLU 前馈层；
- `decoder/lm_head.py`：最终 LayerNorm 与词表投影；
- `model.py`：只负责把这些块顺序组装起来。

预训练入口采用 `torchrun` 环境变量接入 DDP，并从 memory-mapped 的二进制分片中随机截取 `sequence_length + 1` 个 token；前 N 个是输入，后 N 个是 next-token target。语言模型优化器在 `language_model/training/optimizer.py` 中实现了一个小型 Muon/Adam 风格更新器，而不是引入训练器框架。

### 3. 后训练不是一个黑盒按钮

`language_model/scripts/sft.py` 复用预训练模型和 tokenizer，只替换训练数据与损失掩码：

1. 读取预训练 checkpoint 及其模型、数据元数据；
2. 用该 checkpoint 元数据重建同一模型；
3. 读取成对的 input/label 分片；
4. 对 assistant 标签之外的位置使用 `-100`；
5. 保存独立的 SFT checkpoint 和 JSON 元数据。

`language_model/scripts/rl.py` 实现了紧凑的 GRPO：对每个规划题采样一个 completion group，按方案、顺序、预计时长、风险四个最终字段打分，组内标准化 reward；随后以旧策略 ratio 裁剪和冻结 SFT reference 的 KL 项更新策略。奖励不读取或要求 CoT，completion 只是一行可执行的最终计划。

可复现的小型完整实验使用北岭观测站数据，三个 YAML 都从同一 tokenizer 路径读取：

```bash
.venv/bin/python -m language_model.scripts.make_observatory_data
./language_model/scripts/prepare_data.sh pretrain language_model/config/observatory_pretrain.yaml
.venv/bin/python -m language_model.scripts.pretrain language_model/config/observatory_pretrain.yaml
./language_model/scripts/prepare_data.sh sft language_model/config/observatory_sft.yaml
.venv/bin/python -m language_model.scripts.sft language_model/config/observatory_sft.yaml
.venv/bin/python -m language_model.scripts.rl language_model/config/observatory_rl.yaml
```

### 4. 推理与可复现性

`language_model/infer.py` 的职责很窄：加载 `safetensors` 权重及同名 JSON 元数据，按最近的上下文窗口前向计算，然后通过 temperature 和可选 top-k 采样下一个 token。生成会在 tokenizer 中存在的 `<|eos|>` 或 `<|im_end|>` 处停止。

checkpoint、原始数据和编码分片都被 `.gitignore` 排除。克隆仓库不会下载数 GB 的语料或权重；请使用自己的数据、保存自己的实验产物，并保留 tokenizer 与 metadata，二者同样是模型的一部分。

## 视觉模型：两条可运行的 ViT 路径

`vision_model/` 与语言模型独立，提供从随机初始化到预训练微调的两个分类工作流：

```text
Imagenette ImageFolder
  -> 标准 ViT: patch embedding + class token + learned position + Transformer blocks
  -> 从头训练 checkpoint

Imagenette ImageFolder
  -> Hugging Face ViT Base + ClassificationHead
  -> full fine-tune checkpoint
```

`model/encoder/backbone.py` 同时承载两个明确边界：`VisionTransformer` 是从头训练的标准 ViT；`PretrainedVisionBackbone` 只负责加载 Hugging Face 视觉模型并返回 pooled 特征，供 `finetune.py` 全量微调。二者复用同一个短分类头、AdamW 参数分组、训练循环和推理入口，但不共享隐藏的训练框架。

默认数据集是 100 张 Imagenette 图像组成的 10 类 ImageNet 子集。数据脚本通过 HF 镜像下载归档，并固定每类 8 张训练、2 张验证：

```bash
./vision_model/scripts/prepare_data.sh
./vision_model/scripts/train.sh
./vision_model/scripts/infer.sh \
  --checkpoint vision_model/checkpoints/vit_imagenette.pt \
  --image vision_model/data/imagenette_100/val/0/8.jpg
```

预训练微调脚本设置 `HF_ENDPOINT=https://hf-mirror.com`、禁用 Xet，并只下载 `config.json`、image processor 与 `model.safetensors`：

```bash
./vision_model/scripts/finetune.sh
./vision_model/scripts/infer.sh \
  --checkpoint vision_model/checkpoints/vit_base_imagenette_finetune.pt \
  --image vision_model/data/imagenette_100/val/0/8.jpg
```

`train.yaml` 控制从头 ViT 的图像尺寸、patch、宽度、层数和 checkpoint；`finetune.yaml` 控制本地预训练模型、是否冻结骨干、学习率和 checkpoint。两份配置均使用标准 `ImageFolder` 结构：

```text
vision_model/data/imagenette_100/
  train/0/ ... train/9/
  val/0/   ... val/9/
```

## 安装

项目使用 Python 3.10+ 的类型语法，建议在独立虚拟环境中安装。PyTorch 需要选择与你的 CPU/CUDA 环境匹配的 wheel；其余依赖由 `requirements.txt` 固定范围。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 的激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

安装后，可直接生成并编码这条小型语言模型链路：

```bash
.venv/bin/python -m language_model.scripts.make_observatory_data
```

## 当前执行边界

北岭观测站实验刻意是一个小领域过拟合链路，用来检查数据、tokenizer、预训练、SFT 和 GRPO 的契约，而不是通用知识问答基准。扩大模型或语料前，应保持 tokenizer 与已编码数据、checkpoint 的同一版本关系。

## 目录地图

```text
.
├── language_model/
│   ├── config/       # pretrain / SFT / RL 参数
│   ├── model/        # tokenizer、decoder、RoPE、LM head
│   ├── scripts/      # 数据准备、预训练、SFT、RL 入口
│   ├── training/     # loss 与优化器
│   ├── checkpoints/   # 权重与 tokenizer
│   └── infer.py
├── vision_model/
│   ├── config/       # from-scratch / fine-tune 配置
│   ├── model/        # patch tokenizer、ViT、HF backbone、分类头
│   ├── scripts/      # 下载、训练、微调、推理
│   ├── training/     # ImageFolder、loop、loss、optimizer
│   └── infer.py
├── requirements.txt
└── LICENSE
```

## 贡献原则

保持项目小，不等于牺牲严谨性。提交代码时请遵守这些简单规则：

1. 先复用已有模块、PyTorch 标准能力或 Python 标准库；不要为了一个入口引入框架。
2. 一个新概念对应一个短文件，优先把现有超 100 行的编排代码拆小。
3. 数据格式、checkpoint 格式和 tokenizer 变更必须一起更新元数据校验。
4. 新增非平凡逻辑时，留下一个最小的可运行 shape/行为检查。
5. 没有实际需求的 registry、factory、回调系统和“以后可能会用”的配置项，先不要写。

## 许可证

本项目采用 [MIT License](LICENSE)。
