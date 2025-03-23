import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.empty((1, max_len, embed_dim)))
        init.xavier_uniform_(self.pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(F.gelu(self.fc1(x)) + x)
        x = self.norm2(self.fc2(x) + x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        nhead=8,
        nlayer=6,
        ff_dim=512,
        channels=1,
        patch_size=8,
        num_classes=10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cls_emb = self._cls_emb(embed_dim)
        self.emb = nn.Linear(channels * patch_size**2, embed_dim)
        self.pe = PositionalEncoding(embed_dim)

        # encoder layers use GELU as described in the paper
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, ff_dim, activation=F.gelu, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, nlayer)
        self.mlp = MLP(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def _pad(self, x):
        H, W = x.shape[-2:]
        p = self.patch_size
        W_n = p * math.ceil(W / p)
        H_n = p * math.ceil(H / p)

        pad = (0, W_n - W, 0, H_n - H)
        x = F.pad(x, pad, mode="constant", value=0)
        return x

    def _resize(self, x):
        H, W = x.shape[-2:]
        p = self.patch_size
        N_w, N_h = math.ceil(W / p), math.ceil(H / p)
        x = F.interpolate(x, (N_w * p, N_h * p), mode="nearest")
        return x

    def _split_into_patches(self, x):
        N = x.size(-1) * x.size(-2) // self.patch_size**2
        x = x.reshape((x.size(0), N, -1))
        return x

    def _cls_emb(self, embed_dim):
        cls_emb = nn.Parameter(torch.empty((1, 1, embed_dim)))
        init.xavier_uniform_(cls_emb)
        return cls_emb

    def forward(self, x):
        # pad so that the shape is divisible by patch_size.
        # you can also resize instead, but this yields similar results
        x = self._pad(x)
        # x = self._resize(x)

        # split into N patches and flatten,
        # getting (B, N, HW/P^2) shaped tensor
        x = self._split_into_patches(x)

        x = self.emb(x)

        # prepend each image with a class embedding
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat([cls_emb, x], dim=1)

        x = self.pe(x)
        x = self.encoder(x)
        x = self.mlp(x)

        # classification head connected to the class embedding
        out = self.head(x[:, 0, :])
        return out
