# nanoGPT

A small, inspectable ViT teaching framework for image classification. The
model code keeps patch tokenization, two-dimensional RoPE, Q/K/V attention,
the GELU MLP, a pretrained vision wrapper, and a linear probe in separate
modules.

The first training path is deliberately narrow: it freezes a pretrained
vision encoder and trains one `nn.Linear` classification head with AdamW.
There is no activation or optimizer registry in this first version. The
from-scratch `VisionTransformer` remains available for reading and shape
experiments; its patch tokens use the same 2D RoPE implementation as the
attention module.

## Install

Use Python 3.11 or newer from this directory. Install a PyTorch wheel that
matches the target CUDA version when a GPU is available.

```powershell
cd D:\python\nanoGPT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download SigLIP2 So400M

The approximately 400M-parameter SigLIP2 vision checkpoint is hosted at:

<https://huggingface.co/google/siglip2-so400m-patch14-384>

After installing `huggingface_hub`, download it locally with:

```powershell
hf download google/siglip2-so400m-patch14-384 `
  --local-dir vision_model/models/siglip2-so400m-patch14-384
```

The older equivalent command is:

```powershell
huggingface-cli download google/siglip2-so400m-patch14-384 `
  --local-dir vision_model/models/siglip2-so400m-patch14-384
```

Set `model.name` in `vision_model/config/train.yaml` to
`vision_model/models/siglip2-so400m-patch14-384` for an offline/local run. Leaving the Hub
ID in the YAML lets Transformers download the files on first use.

## Dataset Layout

The trainer uses `torchvision.datasets.ImageFolder` and expects matching class
directories in the training and validation splits:

```text
vision_model/data/
  train/
    class_a/
    class_b/
  validation/
    class_a/
    class_b/
```

## Train And Infer

```powershell
python -m vision_model.training.train --config vision_model/config/train.yaml
python -m vision_model.infer `
  --checkpoint vision_model/checkpoints/siglip2_linear_probe.pt `
  --image vision_model/data/validation/class_a/example.jpg
```

The checkpoint stores the backbone ID, revision, hidden width, class names,
and the probe weights. `seed: 7` and `num_workers: 0` are set in the example
configuration so that the data order and initialization can be reproduced.

## Read The Model

`vision_model/model/tokenizer.py` turns an image into continuous patch vectors; it does not
look up a learned vocabulary. `vision_model/model/encoder/rope.py` assigns row and column
coordinates to those patches and rotates only Q and K. The attention module
imports that implementation instead of duplicating the rotation equations.
`vision_model/model/heads/classification.py` mean-pools the returned tokens and applies the
single trainable linear probe.
