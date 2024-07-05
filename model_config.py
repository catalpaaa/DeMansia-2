import torch
from dataclasses import dataclass, field


@dataclass
class DeMansia_2_tiny_config:
    # batch size 11024
    learning_rate: float = 1e-4
    weight_decay: float = 0.5
    warmup_epochs:int = 10
    img_size: tuple[int] = (224, 224)
    patch_size: tuple[int] = (16, 16)
    token_label_size: int = 14
    channels: int = 3
    depth: int = 24
    d_model: int = 192
    d_intermediate: int = 0
    num_classes: int = 1000
    ssm_cfg: dict = field(
        default_factory=lambda: {
            "headdim": 48,
        }
    )
    attn_layer_idx: tuple[int] = (10, 20)
    attn_cfg: dict = field(
        default_factory=lambda: {
            "causal": False,
            "d_conv": 4,
            "head_dim": 96,
            "num_heads": 3,
            "out_proj_bias": True,
            "qkv_proj_bias": True,
            "rotary_emb_dim": 48,
        }
    )
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


@dataclass
class DeMansia_2_small_config:
    # batch size 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.5
    warmup_epochs: int = 10
    img_size: tuple[int] = (224, 224)
    patch_size: tuple[int] = (16, 16)
    token_label_size: int = 14
    channels: int = 3
    depth: int = 24
    d_model: int = 384
    d_intermediate: int = 0
    num_classes: int = 1000
    ssm_cfg: dict = field(
        default_factory=lambda: {
            "headdim": 96,
        }
    )
    attn_layer_idx: tuple[int] = (10, 20)
    attn_cfg: dict = field(
        default_factory=lambda: {
            "causal": False,
            "d_conv": 4,
            "head_dim": 96,
            "num_heads": 6,
            "out_proj_bias": True,
            "qkv_proj_bias": True,
            "rotary_emb_dim": 48,
        }
    )
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
