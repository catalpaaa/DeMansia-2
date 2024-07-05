import copy
import math
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from timm.models.layers import lecun_normal_, trunc_normal_
from torchmetrics.classification import Accuracy

from bidirectional_mamba_2 import Bidirectional_Mamba_2
from modules.data import create_token_label_target
from modules.lr_scheduler import LinearWarmupCosineAnnealingLR
from modules.optimizer import Lion
from modules.token_ce import TokenLabelCrossEntropy


# https://github.com/zihangJiang/TokenLabeling/blob/main/tlt/models/layers.py#L316
class PatchEmbed4_2(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(
        self,
        img_size: tuple[int] = (224, 224),
        patch_size: tuple[int] = (16, 16),
        in_chans=3,
        d_model=768,
    ):
        super().__init__()

        new_patch_size = (patch_size[0] // 2, patch_size[1] // 2)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = d_model

        self.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(
            64, d_model, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg={},
    attn_layer_idx=[],
    attn_cfg={},
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg)
        if ssm_cfg.pop("layer", None):
            print(
                '\033[91mViM 2 uses mamba 2 only, setting "layer" in ssm config will not change anything.\033[0m'
            )

        mixer_cls = partial(
            Bidirectional_Mamba_2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs,
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx

    return block


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class DeMansia_2(pl.LightningModule):
    def __init__(
        self,
        learning_rate=7e-3,
        weight_decay=0.15,
        warmup_epochs=10,
        img_size=(224, 224),
        patch_size=(16, 16),
        token_label_size=14,
        channels=3,
        depth=24,
        d_model=192,
        d_intermediate=0,
        num_classes=1000,
        ssm_cfg={},
        attn_layer_idx=[],
        attn_cfg={},
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.token_label_size = token_label_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        self.patch_embed = PatchEmbed4_2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            d_model=d_model,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embed.num_patches + 1,
                self.d_model,
            )
        )

        self.head = nn.Linear(self.d_model, num_classes)
        self.aux_head = nn.Linear(self.d_model, num_classes)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.token_loss = TokenLabelCrossEntropy(
            dense_weight=1.0,
            cls_weight=1.0,
            mixup_active=False,
            classes=self.num_classes,
            ground_truth=False,
        )
        self.token_loss.to(self.device)

        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        self.aux_head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.valid_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    # modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def forward_features(
        self,
        hidden_states,
        inference_params=None,
    ):
        B, M, _ = hidden_states.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        self.token_position = M // 2

        hidden_states = torch.cat(
            (
                hidden_states[:, : self.token_position, :],
                cls_token,
                hidden_states[:, self.token_position :, :],
            ),
            dim=1,
        )
        hidden_states = hidden_states + self.pos_embed
        M = hidden_states.shape[1]

        # mamba impl
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states

    def forward(
        self,
        x,
        inference_params=None,
        return_features=False,
    ):
        x = self.patch_embed(x)

        if return_features is False:
            if self.training:
                lam = np.random.beta(1.0, 1.0)
                patch_h, patch_w = x.shape[2], x.shape[3]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                temp_x = x.clone()
                temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[
                    :, :, bbx1:bbx2, bby1:bby2
                ]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_features(
            x,
            inference_params,
        )

        if return_features:
            return x

        x_cls = self.head(x[:, self.token_position, :])
        x_aux = self.aux_head(
            torch.cat(
                [x[:, : self.token_position, :], x[:, self.token_position + 1 :, :]],
                dim=1,
            )
        )

        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]

        x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
        temp_x = x_aux.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
        x_aux = temp_x
        x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def training_step(self, batch):
        self.training = True

        sample, target = batch
        preds = self(sample)

        target = create_token_label_target(
            target,
            num_classes=self.num_classes,
            smoothing=0.1,
            device=self.device,
            label_size=self.token_label_size,
        )
        loss = self.token_loss(preds, target)

        self.log(
            "Training Loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.training = False
        return loss

    def validation_step(self, batch):
        sample, target = batch
        preds = self(sample)

        loss = self.ce_loss(preds, target)

        self.log(
            "Validation Accuracy Top 1",
            self.valid_acc_top_1(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation Accuracy Top 5",
            self.valid_acc_top_5(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation Loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self):
        pl.utilities.memory.garbage_collection_cuda()

    def on_validation_epoch_end(self):
        pl.utilities.memory.garbage_collection_cuda()

    def configure_optimizers(self):
        optimizer = Lion(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler_config = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=300,
                warmup_start_lr=1e-7,
                eta_min=1e-6,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "Learning Rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
