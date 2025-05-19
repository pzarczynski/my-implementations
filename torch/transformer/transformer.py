import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qw = nn.Linear(embed_dim, embed_dim)
        self.kw = nn.Linear(embed_dim, embed_dim)
        self.vw = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask=None,
        padding_mask=None,
    ):
        B = q.size(0)

        # split input into heads and transpose
        # so that the shape is (B, H, L, D_h)
        Q = self.qw(q).reshape(B, -1, self.num_heads, self.head_dim)
        K = self.kw(k).reshape(B, -1, self.num_heads, self.head_dim)
        V = self.vw(v).reshape(B, -1, self.num_heads, self.head_dim)

        Q, K, V = [t.transpose(1, 2) for t in [Q, K, V]]

        # calculate query-key similarities with
        # 1 / sqrt(D_h) factor for numerical stability
        attn_scores = (Q @ K.transpose(-1, -2)) / (self.head_dim**0.5)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, -torch.inf)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -torch.inf)

        attn_probs = F.softmax(attn_scores, dim=-1)

        # after applying masks, some sequences contain
        # only -inf, resulting in NaNs in softmax fn
        attn_probs = attn_probs.nan_to_num()

        # for each embedding, we calculate
        # the sum of its attending values
        context = attn_probs @ V

        # (B, H, L, D_h) -> (B, L, D)
        context = context.transpose(1, 2).reshape(q.shape)

        # out projection as described in paper
        out = self.out_proj(context)
        return out


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_padding_mask):
        attn = self.attn(x, x, x, padding_mask=src_padding_mask)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(
            EncoderLayer(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        )

    def forward(self, x, src_padding_mask):
        for layer in self.layers:
            x = layer(x, src_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, context, attn_mask, src_padding_mask, tgt_padding_mask):
        attn = self.self_attn(x, x, x, attn_mask, tgt_padding_mask)
        x = self.norm1(x + attn)

        attn = self.cross_attn(x, context, context, padding_mask=src_padding_mask)
        x = self.norm2(x + attn)

        x = self.norm3(x + self.ffn(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(
            DecoderLayer(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        )

    def forward(self, x, context, attn_mask, src_padding_mask, tgt_padding_mask):
        for layer in self.layers:
            x = layer(x, context, attn_mask, src_padding_mask, tgt_padding_mask)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder = Encoder(embed_dim, hidden_dim, num_heads, num_layers)
        self.decoder = Decoder(embed_dim, hidden_dim, num_heads, num_layers)

    def _casual_mask(self, size):
        return torch.tril(torch.ones(size, size))

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        context = self.encoder(src, src_padding_mask)

        # casual mask prevents future keys from attending to queries
        attn_mask = self._casual_mask(tgt.size(1)).to(tgt.device)

        out = self.decoder(tgt, context, attn_mask, src_padding_mask, tgt_padding_mask)
        return out
