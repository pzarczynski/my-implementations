import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from tqdm import tqdm

from .transformer import Transformer

EPOCHS = 5
TRAIN_SIZE = 10000
VAL_SIZE = 200

EMBED_DIM = 128
HIDDEN_DIM = 256
NHEAD = 8
NLAYERS = 6
BATCH_SIZE = 128
SEQ_LEN = 16
VOCAB_SIZE = 128


def generate_examples(n, voc_size, seq_len):
    """generates example data - random
    sequences and their reversed versions"""
    srcs, tgts = [], []

    for _ in range(n):
        src = [
            random.randint(3, voc_size - 1) for _ in range(random.randint(0, seq_len))
        ]

        tgt = torch.tensor([1] + src[::-1] + [2], dtype=torch.int64)
        tgts.append(tgt)

        src = torch.tensor(src, dtype=torch.int64)
        srcs.append(src)

    srcs = pad_sequence(srcs, batch_first=True)
    tgts = pad_sequence(tgts, batch_first=True)
    return srcs, tgts


def validate_model(model, loader, device, num_classes):
    accuracy = Accuracy(task="multiclass", ignore_index=0, num_classes=num_classes)
    accuracy = accuracy.to(device)

    model.eval()

    for src, tgt in tqdm(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_out = tgt[:, 1:]

        with torch.no_grad():
            output = model.generate(src)

        accuracy.update(output, tgt_out)

    return accuracy.compute()


class TransformerModel(nn.Module):
    """example transformer architecture."""

    def __init__(self, embed_dim, hidden_dim, nhead, nlayers, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.transformer = Transformer(embed_dim, hidden_dim, nhead, nlayers)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_padding_mask = src == 0
        tgt_padding_mask = tgt == 0

        src = self.emb(src)
        tgt = self.emb(tgt)

        out = self.transformer(src, tgt, src_padding_mask, tgt_padding_mask)
        out = self.out_proj(out)
        return out

    def generate(self, src):
        """generate output in an autoregressive manner."""
        B, L = src.shape
        device = src.device

        src_padding_mask = src == 0

        src_emb = self.emb(src)
        src_emb = self.transformer.positional_encoding(src_emb)

        context = self.transformer.encoder(src_emb, src_padding_mask)

        # initial sequence, containing only <SOS> tokens
        tgt = torch.ones(B, 1, dtype=torch.int64).to(device)
        tgt_emb = self.emb(tgt)

        for i in range(L + 1):
            tgt_emb = self.transformer.positional_encoding(tgt_emb)

            # I think attn_mask is not needed here, as the model generates output
            # tokens one-by-one and cannot 'peek' at future tokens
            # but the model shows better performance with it, probably due to
            # numerical stability or some other phenomenon I can't yet understand
            attn_mask = self.transformer._causal_mask(tgt.size(1)).to(src.device)

            # tgt_padding_mask isn't needed as <EOS> token appears prior to the padding
            preds = self.transformer.decoder(
                tgt_emb, context, attn_mask, src_padding_mask, None
            )
            preds = preds[:, -1, :].unsqueeze(1)

            next_token = self.out_proj(preds)
            next_token = torch.argmax(next_token, dim=-1)

            tgt = torch.cat([tgt, next_token], dim=-1)
            tgt_emb = self.emb(tgt)

        return tgt[:, 1:]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src, tgt = generate_examples(TRAIN_SIZE + VAL_SIZE, VOCAB_SIZE, SEQ_LEN)

    train_src, train_tgt = src[:TRAIN_SIZE], tgt[:TRAIN_SIZE]
    val_src, val_tgt = src[TRAIN_SIZE:], tgt[TRAIN_SIZE:]

    train_dataset = TensorDataset(train_src, train_tgt)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_src, val_tgt)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = TransformerModel(EMBED_DIM, HIDDEN_DIM, NHEAD, NLAYERS, VOCAB_SIZE)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    epoch_loss = []
    epoch_acc = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, (src, tgt) in enumerate(tqdm(train_loader)):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        epoch_loss.append(train_loss)

        val_acc = validate_model(model, val_loader, device, VOCAB_SIZE).cpu()
        print(f"Epoch {epoch+1} val accuracy: {val_acc}")
        epoch_acc.append(val_acc.cpu())

    plt.plot(epoch_loss, label="loss")
    plt.plot(epoch_acc, label="accuracy")
    plt.xlabel("epochs")

    plt.legend()
    plt.savefig("training_curve.png")
