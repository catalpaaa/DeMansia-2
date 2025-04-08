# DeMansia 2

## About

DeMansia 2 introduces Mamba 2 to the realm of computer vision, with performance improvements from bidirectional Mamba 2 and token labeling training.

## Installation

We provided a simple [setup.sh](setup.sh) to install the Conda environment. You need to satisfy the following prerequisite:

- Linux
- NVIDIA GPU
- CUDA 12+ supported GPU driver
- Miniforge

Then, simply run `source ./setup.sh` to get started.

## Pretrained Models

These models were trained on the [ImageNet-1k dataset](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) using a single RTX 4090 during our experiments.

Currently, only DeMansia 2 Tiny is available. We will release more models as opportunities arise and continue to improve current models as our training methods advance.

| Name            | Model Dim. | Num. of Layers | Num. of Param. | Input Res. | Top-1 | Top-5 | Batch Size | Download              | Training Log    |
|-----------------|------------|----------------|----------------|------------|-------|-------|------------|-----------------------|-----------------|
| DeMansia 2 Tiny | 192        | 24             | 9.5M           | 224²       | 79.5% | 94.4% | 1024       | [link][tiny download] | [log][tiny log] |

[tiny download]: https://archive.org/download/DeMansia-2/DeMansia%202%20Tiny%20EMA.ckpt
[tiny log]: https://wandb.ai/catalpa/DeMansia%202%20Tiny/runs/5pip6hjg

## Training and inferencing

To set up the ImageNet-1k dataset, download both the training and validation sets. Use this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to extract and organize the dataset. You should also download and extract the token labeling dataset from [here](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view?usp=sharing).

We provide [train.py](train.py), which contains all the necessary code to train a DeMansia 2 model and log the training progress. The logged parameters can be modified in [model.py](model.py).

The base model's hyperparameters are stored in [model_config.py](model_config.py), and you can adjust them as needed. When further training our model, note that all hyperparameters are saved directly in the model file. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint). The same applies to inferencing, as PyTorch Lightning automatically handles all parameters when loading our model.

Here's a sample code snippet to perform inferencing with DeMansia 2:

```python
import torch

from model import DeMansia_2

model = DeMansia_2.load_from_checkpoint("path_to.ckpt")
model.eval()

sample = torch.rand(3, 224, 224) # Channel, Width, Height
sample = sample.unsqueeze(0) # Batch, Channel, Width, Height
pred = model(sample) # Batch, # of class
```

## Credits

Our work builds upon the remarkable achievements of [Mamba](https://arxiv.org/abs/2312.00752), and [LV-ViT](https://arxiv.org/abs/2104.10858).

[module/data](modules/data) and [module/token_ce.py](module/token_ce.py) are modified from the [LV-ViT repo](https://github.com/zihangJiang/TokenLabeling).

[module/ema](modules/ema) is modified from [here](https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/utils/__init__.py).

[modules/optimizer.py](modules/optimizer.py) is taken from [here](https://github.com/google/automl/blob/master/lion/lion_pytorch.py).

Check out the original DeMansia [here](https://github.com/catalpaaa/DeMansia).
