<h1 align="center">NanoAgent</h1>

<p align="center"><strong>从原始语料到后训练，把 Agent 框架搭建与底层基模训练的完整链路收敛为最简形式的文件目录。</strong></p>

<p align="center">
  一个为理解而建的视觉-语言模型实现：不靠巨型训练框架，<br />
  从 Byte-level BPE、token 分片、预训练、SFT 到采样生成，**每一步都能追到代码。**
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

预训练使用 `language_model/tokenizer_model/` 中的 byte-level BPE 词表；默认配置的词表大小是 3,200。`language_model/model/tokenizer.py` 还会为 vocab 和 merges 计算 SHA-256 指纹，使 checkpoint 能拒绝不匹配的 tokenizer。

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
2. 验证 SFT 词表大小与预训练阶段一致；
3. 读取成对的 input/label 分片；
4. 对 assistant 标签之外的位置使用 `-100`；
5. 保存独立的 SFT checkpoint 和 JSON 元数据。

这让“预训练”和“后训练”成为两种可检查的目标函数，而不是两套互不相干的项目。`language_model/scripts/rl.py` 与 `language_model/config/rl.yaml` 目前是预留位置，**尚未实现 RL/RLHF**；README 不把空文件包装成已交付能力。

### 4. 推理与可复现性

`language_model/infer.py` 的职责很窄：加载 `safetensors` 权重及同名 JSON 元数据，校验 tokenizer 指纹，按最近的上下文窗口前向计算，然后通过 temperature 和可选 top-k 采样下一个 token。

接口形状如下：

```bash
python -m language_model.infer \
  --model language_model/checkpoints/transformer_sft.safetensors \
  --tokenizer-dir language_model/tokenizer_model \
  --prompt "你好，介绍一下 Transformer" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

checkpoint、原始数据和编码分片都被 `.gitignore` 排除。克隆仓库不会下载数 GB 的语料或权重；请使用自己的数据、保存自己的实验产物，并保留 tokenizer 与 metadata，二者同样是模型的一部分。

## 视觉支线：同样的可读性原则

`vision_model/` 是一条独立的视觉实验路径，不是语言模型训练的依赖。它展示同一个原则如何用于 ViT：

```text
ImageFolder
  -> Hugging Face image processor
  -> frozen pretrained vision backbone
  -> mean pool patch tokens
  -> one LinearProbe
  -> class probabilities
```

代码中同时保留了可阅读的 ViT 教学部件和预训练骨干的线性探针封装：

- `model/tokenizer.py`：图像切块与连续 patch embedding；
- `model/encoder/rope.py`：二维 RoPE；
- `model/encoder/attention.py`、`layers.py`、`backbone.py`：显式 ViT 结构；
- `model/heads/classification.py`：均值池化后的一层线性分类头；
- `training/optimizer.py`：只为 probe 构造 AdamW；
- `config/train.yaml`：ImageFolder 目录、SigLIP2 骨干、训练参数和 checkpoint 路径。

视觉数据目录遵循 `torchvision.datasets.ImageFolder` 的标准布局：

```text
vision_model/data/
  train/
    class_a/
    class_b/
  validation/
    class_a/
    class_b/
```

这条支线目前只定义线性探针训练；`vision_model/scripts/finetune.py` 和对应配置仍为空，不应被理解为全量视觉微调已经完成。

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

安装后，可以先确认语言模型推理接口的参数：

```bash
python -m language_model.infer --help
```

## 当前执行边界

本仓库当前更接近一份正在收口的、可阅读的全链路实现，而不是已经发布的一键训练包。流程、数据格式和模型部件都已落在对应目录中，但在把它用于新训练前应先完成以下接线：

- 预训练和 SFT 入口仍引用了不存在的顶层 `training` 导入，需统一到 `language_model/training/` 的实际 API；
- 预训练 checkpoint 写入格式与推理端要求的 `safetensors` 路径需要统一；
- 视觉训练/推理入口依赖尚未落盘的 `vision_model.settings`，本地安装依赖后还需补齐这层配置接线；
- RL 与视觉全量微调仍是空占位，不是“已支持但未文档化”的功能。

这些边界写在这里，是为了让下一位贡献者从真实状态出发：先让一条小链路通过，再扩大数据、模型或训练规模。

## 目录地图

```text
.
├── language_model/
│   ├── config/       # pretrain / SFT / RL 参数
│   ├── model/        # tokenizer、decoder、RoPE、LM head
│   ├── scripts/      # 数据准备、预训练、SFT、RL 入口
│   ├── training/     # loss 与优化器
│   ├── tokenizer_model/
│   └── infer.py
├── vision_model/
│   ├── config/       # 线性探针配置
│   ├── model/        # patch tokenizer、2D RoPE、ViT、分类头
│   ├── scripts/
│   ├── training/
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
