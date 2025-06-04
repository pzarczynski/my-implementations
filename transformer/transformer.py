import jax
import jax.numpy as jnp
import flax.linen as nn
import chex

from flax.struct import dataclass, field


@dataclass
class TransformerLayerConfig:
    """Config for managing layer hyperparameters."""
    num_heads: int = 4
    head_dim: int = 16
    mlp_factor: float = 4.0
    mlp_dim: int = field(default=None)
    emb_dim: int = field(default=None)
    dropout_rate: float = 0.2
    use_bias: bool = False
    activation = "silu"
    
    def __post_init__(self):
        object.__setattr__(self, "emb_dim", self.num_heads * self.head_dim)
        object.__setattr__(self, "mlp_dim", int(self.emb_dim * self.mlp_factor))
        
        
@dataclass
class TransformerConfig:
    """Config for managing transformer hyperparameters."""
    layer_config: TransformerLayerConfig = TransformerLayerConfig()
    emb_dim: int = field(default=None)
    num_layers: int = 2
    use_bias: bool = False
    vocab_size: int = 10
    max_len: int = 10
    pad_token: int = 0
    sos_token: int = 1
    eos_token: int = 2
    
    def __post_init__(self):
        object.__setattr__(self, "emb_dim", self.layer_config.emb_dim)


class MHA(nn.Module):
    config: TransformerLayerConfig

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        ctx: chex.Array | None = None,
        mask: chex.Array | None = None,
        eval: bool = False
    ) -> chex.Array:
        """Perform multi-head attention on `x` queries and 
        `ctx` keys and values. If `ctx` is `None`, `ctx` = `x`."""
        cfg = self.config
        
        if ctx is None:
            ctx = x

        B = x.shape[:-2]

        q = nn.Dense(cfg.emb_dim, use_bias=cfg.use_bias, name="q_proj")(x)
        k = nn.Dense(cfg.emb_dim, use_bias=cfg.use_bias, name="k_proj")(ctx)
        v = nn.Dense(cfg.emb_dim, use_bias=cfg.use_bias, name="v_proj")(ctx)

        q, k, v = map(
            lambda t: t
                .reshape(*B, -1, cfg.num_heads, cfg.head_dim)
                .swapaxes(-2, -3),
            (q, k, v),
        )
        attn_logits = jnp.einsum("...qd,...kd->...qk", q, k) # [*B, H, L, L] 
        attn_logits *= cfg.head_dim ** -0.5

        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = jnp.nan_to_num(attn_weights)
        attn_weights = nn.Dropout(cfg.dropout_rate)(attn_weights, deterministic=eval)

        attn_output = attn_weights @ v
        attn_output = attn_output.swapaxes(-2, -3).reshape(x.shape)

        out = nn.Dense(
            features=cfg.emb_dim, 
            use_bias=cfg.use_bias, 
        )(attn_output)
        return out


class MLP(nn.Module):
    config: TransformerLayerConfig
    
    def setup(self):
        cfg = self.config
        if cfg.activation == "relu":
            self.activation = nn.relu
        elif cfg.activation == "silu":
            self.activation = nn.silu
        elif cfg.activation == "gelu":
            self.activation = nn.gelu
        else:
            raise ValueError("Activation must be one of ['relu', 'silu', 'gelu'],"
                             f"not {cfg.activation}.")

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        cfg = self.config
        
        x = nn.Dense(cfg.mlp_dim, use_bias=cfg.use_bias)(x)
        x = self.activation(x)
        x = nn.Dense(cfg.emb_dim, use_bias=cfg.use_bias)(x)
        return x


class TransformerEncoderLayer(nn.Module):
    config: TransformerLayerConfig

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        pad_mask: chex.Array | None = None,
        eval: bool = False,
    ):
        cfg = self.config
        
        x = nn.LayerNorm(use_bias=cfg.use_bias, use_scale=False)(embeddings)
        x = MHA(cfg)(x, mask=pad_mask, eval=eval)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=eval)
        embeddings += x

        x = nn.LayerNorm(use_bias=cfg.use_bias, use_scale=False)(embeddings)
        x = MLP(cfg)(x)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=eval)
        embeddings += x
        return embeddings
    

class TransformerDecoderLayer(nn.Module):
    config: TransformerLayerConfig

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        context: chex.Array | None = None,
        attn_mask: chex.Array | None = None,
        src_pad_mask: chex.Array | None = None,
        eval: bool = False,
    ):
        cfg = self.config
        
        x = nn.LayerNorm(use_bias=cfg.use_bias, use_scale=False)(embeddings)
        x = MHA(cfg)(x, mask=attn_mask, eval=eval)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=eval)
        embeddings += x

        x = nn.LayerNorm(use_bias=cfg.use_bias, use_scale=False)(embeddings)
        x = MHA(cfg)(x, ctx=context, mask=src_pad_mask, eval=eval)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=eval)
        embeddings += x

        x = nn.LayerNorm(use_bias=cfg.use_bias, use_scale=False)(embeddings)
        x = MLP(cfg)(x)
        x = nn.Dropout(cfg.dropout_rate)(x, deterministic=eval)
        embeddings += x
        return embeddings

    
class PositionalEncoding(nn.Module):
    """Classic sine-cosine positional encoding."""
    config: TransformerConfig

    def setup(self):
        cfg = self.config
        
        position = jnp.arange(cfg.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, cfg.emb_dim, 2) *
                           -(jnp.log(10000.0) / cfg.emb_dim))

        pe = jnp.zeros((cfg.max_len, cfg.emb_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe * cfg.emb_dim ** 0.5

    def __call__(self, x: chex.Array) -> chex.Array:
        return x + self.pe[None, :x.shape[1], :]


class TransformerEncoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        src: chex.Array,
        pad_mask: chex.Array | None = None, 
        eval: bool = False,
    ):
        cfg = self.config
        
        x = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
        )(src)
        x = PositionalEncoding(cfg)(x)
        
        if pad_mask is not None:
            # Convert padding mask from [B, L] to [B, 1, L, L] format.
            pad_mask = pad_mask[..., None, :, None] & pad_mask[..., None, None, :]

        for _ in range(cfg.num_layers):
            x = TransformerEncoderLayer(cfg.layer_config)(x, pad_mask=pad_mask, eval=eval)

        x = nn.LayerNorm(
            use_bias=cfg.use_bias, 
            use_scale=False
        )(x)
        x = nn.Dense(
            features=cfg.emb_dim,
            use_bias=cfg.use_bias, 
        )(x)
        return x


class TransformerDecoder(nn.Module):
    config: TransformerConfig
    
    @nn.compact
    def __call__(
        self,
        tgt: chex.Array,
        context: chex.Array | None = None,
        src_pad_mask: chex.Array | None = None,
        tgt_pad_mask: chex.Array | None = None,
        eval: bool = False,
    ):
        cfg = self.config

        x = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
        )(tgt)
        x = PositionalEncoding(cfg)(x)
        
        attn_mask = self.make_causal_mask(x)

        if tgt_pad_mask is not None:
            attn_mask &= tgt_pad_mask[..., None, None, :]

        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask[..., None, None, :]

        for _ in range(cfg.num_layers):
            x = TransformerDecoderLayer(cfg.layer_config)(
                x,
                context=context,
                attn_mask=attn_mask,
                src_pad_mask=src_pad_mask,
                eval=eval,
            )
            
        x = nn.LayerNorm(
            use_bias=cfg.use_bias, 
            use_scale=False
        )(x)
        x = nn.Dense(
            features=cfg.vocab_size,
            use_bias=cfg.use_bias, 
        )(x)
        return x
    
    @staticmethod
    def make_causal_mask(x: chex.Array) -> chex.Array:
        """Create a causal mask with an arbitrary batch shape."""
        B = (1,) * len(x.shape[:-2])
        L = x.shape[-2]
        mask = jnp.tril(jnp.ones((*B, 1, L, L), dtype=bool))
        return mask


class Transformer(nn.Module):
    encoder_config: TransformerConfig = TransformerConfig()
    decoder_config: TransformerConfig = TransformerConfig()
    
    def setup(self):
        self.encoder = TransformerEncoder(self.encoder_config)
        self.decoder = TransformerDecoder(self.decoder_config)
    
    @nn.compact
    def __call__(
        self, 
        src: chex.Array, 
        tgt: chex.Array, 
    ) -> chex.Array:
        src_pad_mask = src != self.encoder_config.pad_token
        context = self.encoder(src, pad_mask=src_pad_mask)
        
        tgt_pad_mask = tgt != self.decoder_config.pad_token
        logits = self.decoder(
            tgt,
            context=context,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )
        return logits
    
    def generate(self, src: chex.Array) -> chex.Array:
        """Auto-regressive output generation."""
        cfg = self.decoder_config
        B = src.shape[:-1]
        
        src_pad_mask = src != cfg.pad_token
        context = self.encoder(src, pad_mask=src_pad_mask, eval=True)
        
        tgt = jnp.zeros((*B, cfg.max_len + 1), dtype=jnp.int32)
        tgt = tgt.at[:, 0].set(cfg.sos_token)

        def step_fn(decoder: TransformerDecoder, tgt: chex.Array, i: int) -> tuple[chex.Array, None]:
            logits = decoder(tgt[:, :-1], context, src_pad_mask=src_pad_mask, eval=True)
            next_token = jnp.argmax(logits[:, i], axis=-1)
            tgt = tgt.at[:, i+1].set(next_token)
            return tgt, None
        
        tgt, _ = nn.scan(
            step_fn, variable_broadcast="params"
        )(self.decoder, tgt, jnp.arange(cfg.max_len))
        return tgt[:, 1:]