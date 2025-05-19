import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
import math


def make_causal_mask(x: chex.Array) -> chex.Array:
    B = (1,) * len(x.shape[:-2])
    L = x.shape[-2]
    mask = jnp.tril(jnp.ones((*B, L, L), dtype=bool))
    return mask


class MHA(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        context: chex.Array | None = None,
        mask: chex.Array | None = None,
    ):
        if context is None:
            context = embeddings

        assert self.embed_dim % self.num_heads == 0
        head_dim = self.embed_dim // self.num_heads

        B = embeddings.shape[:-2]

        q = nn.Dense(self.embed_dim, use_bias=False, name="q_proj")(embeddings)
        k = nn.Dense(self.embed_dim, use_bias=False, name="k_proj")(context)
        v = nn.Dense(self.embed_dim, use_bias=False, name="v_proj")(context)

        # (*B, L, D) -> (*B, H, L, D_h)
        q, k, v = map(
            lambda t: t.reshape(*B, -1, self.num_heads, head_dim).swapaxes(-2, -3),
            (q, k, v),
        )
        attn_logits = jnp.einsum("...qd,...kd->...qk", q, k) / (head_dim**0.5)

        if mask is not None:
            mask = mask[..., None, :, :].astype(bool)
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = jnp.nan_to_num(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.swapaxes(-2, -3).reshape(embeddings.shape)

        out = nn.Dense(self.embed_dim, use_bias=False, name="out_proj")(attn_output)
        return out


class MLP(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(inputs.shape[-1])(x)
        return x


class PositionalEncoding(nn.Module):
    embed_dim: int
    max_len: int = 5000

    def setup(self):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim)
        )

        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        self.pe = pe

    def __call__(self, x):
        return x + self.pe[None, :x.shape[1], :]


class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        pad_mask: chex.Array | None = None,
        dropout_eval: bool = False,
    ):
        x = nn.LayerNorm()(embeddings)
        x = MHA(self.embed_dim, self.num_heads)(x, mask=pad_mask)
        residuals = nn.Dropout(self.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals

        x = nn.LayerNorm()(embeddings)
        x = MLP(self.mlp_dim)(x)
        residuals = nn.Dropout(self.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals
        return embeddings


class TransformerEncoder(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    vocab_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        pad_mask: chex.Array | None = None,
        dropout_eval: bool = False,
    ):
        x = PositionalEncoding(self.embed_dim)(embeddings)
        
        if pad_mask is not None:
            pad_mask = pad_mask[..., None, :]

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                self.embed_dim,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
            )(x, pad_mask=pad_mask, dropout_eval=dropout_eval)

        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.embed_dim, use_bias=False, name="out_proj")(x)
        return x


class TransformerDecoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        context: chex.Array | None = None,
        attn_mask: chex.Array | None = None,
        src_pad_mask: chex.Array | None = None,
        dropout_eval: bool = False,
    ):
        x = nn.LayerNorm()(embeddings)
        x = MHA(self.embed_dim, self.num_heads)(x, mask=attn_mask)
        residuals = nn.Dropout(self.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals

        x = nn.LayerNorm()(embeddings)
        x = MHA(self.embed_dim, self.num_heads)(x, context=context, mask=src_pad_mask)
        residuals = nn.Dropout(self.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals

        x = nn.LayerNorm()(embeddings)
        x = MLP(self.mlp_dim)(x)
        residuals = nn.Dropout(self.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals
        return embeddings


class TransformerDecoder(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    vocab_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        context: chex.Array | None = None,
        src_pad_mask: chex.Array | None = None,
        tgt_pad_mask: chex.Array | None = None,
        dropout_eval: bool = False,
    ):
        x = PositionalEncoding(self.embed_dim)(embeddings)
        attn_mask = make_causal_mask(x)

        if tgt_pad_mask is not None:
            tgt_pad_mask = tgt_pad_mask[..., None, :]
            attn_mask &= tgt_pad_mask

        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask[..., None, :]

        for _ in range(self.num_layers):
            x = TransformerDecoderLayer(
                self.embed_dim,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
            )(
                x,
                context=context,
                attn_mask=attn_mask,
                src_pad_mask=src_pad_mask,
                dropout_eval=dropout_eval,
            )

        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.vocab_size, name="out_proj")(x)
        return x
