import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field


@dataclass
class ViTConfig:
    """Config class used for ViT initialization."""
    num_heads: int = 8
    dim_per_head: int = 8
    emb_dim: int = field(default=None)
    num_layers: int = 2
    mlp_factor: float = 4.0
    mlp_dim: int = field(default=None)
    patch_size: int = 4
    activation: str = F.gelu
    dropout_rate: float = 0.1
    use_bias: bool = True
    channels: int = 1
    num_classes: int = 10
    max_len: int = 100
    
    def __post_init__(self):
        object.__setattr__(self, "emb_dim", self.num_heads * self.dim_per_head)
        object.__setattr__(self, "mlp_dim", int(self.emb_dim * self.mlp_factor))
        

class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig = ViTConfig()):
        super().__init__()
        self.patch_size = cfg.patch_size
        
        self.cls_embeds = nn.Embedding(
            num_embeddings=1,
            embedding_dim=cfg.emb_dim
        )
        self.patch_embeds = nn.Conv2d(
            in_channels=cfg.channels, 
            out_channels=cfg.emb_dim, 
            kernel_size=cfg.patch_size, 
            stride=cfg.patch_size
        )
        self.pos_embeds = nn.Embedding(
            num_embeddings=cfg.max_len,
            embedding_dim=cfg.emb_dim
        )
        self.dropout = nn.Dropout(cfg.dropout_rate)
        
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim, 
            nhead=cfg.num_heads, 
            dim_feedforward=cfg.mlp_dim, 
            dropout=cfg.dropout_rate,
            activation = cfg.activation,
            batch_first=True,
            bias=cfg.use_bias,
        )

        self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
        self.cls_norm = nn.LayerNorm(cfg.emb_dim)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.num_classes, bias=cfg.use_bias)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        w_pad = (p - x.size(-1) % p) % p
        h_pad = (p - x.size(-2) % p) % p
        x = F.pad(x, (0, w_pad, 0, h_pad))
        return x

    def forward(self, x):
        # Input is padded so that its shape is divisible by P.
        x = self.pad(x)

        # Data is split into patches and flattened.
        x = self.patch_embeds(x) # B, C, H/P, W/P
        x = x.view(*x.shape[:2], -1).transpose(-1, -2)
        
        # Add a learned positional encoding like in standard transformer.
        x += self.pos_embeds(torch.arange(x.size(-2), device=x.device))

        # Prepend each image with a class embedding.
        cls_embed = self.cls_embeds(torch.arange(1, device=x.device))
        cls_embed = torch.broadcast_to(cls_embed, (*x.shape[:-2], 1, x.shape[-1]))
        x = torch.cat([cls_embed, x], dim=-2)
        x = self.dropout(x)
        
        x = self.encoder(x)
        
        # Classification head is connected to the class embedding.
        x = self.cls_norm(x[:, 0, :])
        x = self.out_proj(x)
        return x
