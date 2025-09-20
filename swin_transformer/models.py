import jax
import flax.linen as nn
import chex
import numpy as np

from jax import random, numpy as jnp
from flax.struct import dataclass, field
from functools import partial
from itertools import product


@dataclass
class SwinTransformerConfig:
    input_size: int = 32
    emb_dim: int = 64
    heads: tuple[int, ...] = (2, 4)
    depths: tuple[int, ...] = (2, 2)
    num_classes: int = 10
    patch_size: int = 2
    window_size: int = 4
    mlp_factor: float = 4.0
    drop_rate: float = 0.1
    stochastic_depth: float = 0.2
    mlp_dim: int = field(default=None)
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "mlp_dim", int(self.emb_dim * self.mlp_factor))


class MLP(nn.Module):
    config: SwinTransformerConfig
    level: int

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool = False) -> chex.Array:
        cfg = self.config
        x = nn.Dense(cfg.mlp_dim * 2**self.level)(x)
        x = nn.gelu(x)
        x = nn.Dropout(cfg.drop_rate)(x, deterministic=eval)
        x = nn.Dense(cfg.emb_dim * 2**self.level)(x)
        x = nn.Dropout(cfg.drop_rate)(x, deterministic=eval)
        return x


class WMSA(nn.Module):
    config: SwinTransformerConfig
    level: int
    
    def setup(self) -> None:
        cfg = self.config
        self.scaled_emb_dim = cfg.emb_dim * 2**self.level
        
        num_pos = 2 * cfg.window_size - 1
        self.bias_table = self.param(
            'bias_table',
            nn.initializers.truncated_normal(0.02),
            (cfg.heads[self.level], num_pos, num_pos)
        )
        
        coords = np.arange(cfg.window_size)
        coords = np.stack(np.meshgrid(coords, coords, indexing='ij'))  # 2, Wh, Ww
        coords = coords.reshape(2, -1) # 2, Wh*Ww
        self.relative_pos = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
        self.relative_pos += cfg.window_size - 1
    
    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array = None, eval: bool = False) -> chex.Array:
        cfg = self.config
        
        x = nn.SelfAttention(
            num_heads=cfg.heads[self.level], 
            qkv_features=self.scaled_emb_dim, 
            dropout_rate=cfg.drop_rate,
            attention_fn=partial(
                nn.dot_product_attention,
                bias=self.bias_table[:, *self.relative_pos],
            )
        )(x, mask=mask, deterministic=eval)
        
        x = nn.Dense(self.scaled_emb_dim)(x)
        x = nn.Dropout(cfg.drop_rate)(x, deterministic=eval)
        return x
    
    
def window_partition(x: chex.Array, size: int) -> chex.Array:
    B, H, W, C = x.shape
    x = x.reshape(B, H // size, size, W // size, size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x


def window_inverse(x: chex.Array) -> chex.Array:
    B, Nh, Nw, H, W, C = x.shape
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, Nh*H, Nw*W, C)
    return x


class StochasticDepth(nn.Module):
    p: float
    
    @nn.compact
    def __call__(self, x: chex.Array, eval: bool) -> chex.Array:
        if eval or self.p == 0.0:
            return x
        key = self.make_rng('stochastic_depth')
        shape = (x.shape[0], *((1,) * (x.ndim - 1)))
        mask = random.bernoulli(key, 1 - self.p, shape)
        return x * mask / (1 - self.p)


class SwinTransformerBlock(nn.Module):
    config: SwinTransformerConfig
    shifted: bool
    level: int
    
    def setup(self) -> None:
        cfg = self.config
        S = cfg.window_size
        R = cfg.input_size // cfg.patch_size // (2**self.level)
        
        if self.shifted:
            shift_mask = jnp.zeros((1, R, R, 1), dtype=jnp.int32)  # 1, H, W, 1
            slices = (slice(0, S), slice(S, S//2), slice(S//2, None))
            
            for i, (s1, s2) in enumerate(product(slices, repeat=2)):
                shift_mask = shift_mask.at[:, s1, s2, :].set(i)

            shift_mask = window_partition(shift_mask, S) # 1, Nh, Nw, S, S, 1
            shift_mask = shift_mask.reshape(*shift_mask.shape[:3], -1) # 1, Nh, Nw, S^2
            shift_mask = shift_mask[..., :, None] == shift_mask[..., None, :] # 1, Nh, Nw, S^2, S^2
            shift_mask = shift_mask[..., None, :, :] # 1, Nh, Nw, 1, S^2, S^2
            self.attn_mask = shift_mask
        else:
            self.attn_mask = None
            
    @nn.compact
    def __call__(self, embeddings: chex.Array, eval: bool) -> chex.Array:
        cfg = self.config
        S = cfg.window_size
        
        x = nn.LayerNorm()(embeddings)
    
        if self.shifted:
            x = jnp.roll(x, shift=S//2, axis=(1, 2))
            
        windows = window_partition(x, S) # B, Nh, Nw, S, S, C
        x = windows.reshape(*windows.shape[:3], -1, windows.shape[-1]) # N, S^2, C
        
        x = WMSA(cfg, self.level)(x, mask=self.attn_mask, eval=eval)
        
        x = x.reshape(windows.shape)
        x = window_inverse(x)
        
        if self.shifted:
            x = jnp.roll(x, shift=-S//2, axis=(1, 2))
            
        embeddings += StochasticDepth(cfg.stochastic_depth)(x, eval=eval)
        x = nn.LayerNorm()(embeddings)
        x = MLP(cfg, self.level)(x, eval=eval)
        embeddings += StochasticDepth(cfg.stochastic_depth)(x, eval=eval)
        return x

        
class SwinTransformer(nn.Module):
    config: SwinTransformerConfig = SwinTransformerConfig()
    
    def patch_embedding(self, x: chex.Array):
        cfg = self.config
        x = nn.Conv(
            features=cfg.emb_dim,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=cfg.patch_size
        )(x)
        return x

    def patch_merging(self, x: chex.Array) -> chex.Array:
        x = jnp.pad(x, ((0, 0), (0, x.shape[1] % 2), (0, x.shape[2] % 2), (0, 0)))
        x = window_partition(x, 2)
        x = x.reshape(*x.shape[:3], -1)
        
        x = nn.LayerNorm()(x)
        x = nn.Dense(x.shape[-1] // 2)(x)
        return x
    
    @nn.compact
    def __call__(self, x: chex.Array, eval: bool = False) -> chex.Array:
        cfg = self.config
        x = self.patch_embedding(x)
        
        for level, depth in enumerate(cfg.depths):
            for i in range(depth):
                x = SwinTransformerBlock(
                    config=cfg, 
                    shifted=bool(i % 2), 
                    level=level
                )(x, eval)
            
            if level < len(cfg.depths) - 1:
                x = self.patch_merging(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(cfg.num_classes)(x)
        return x
