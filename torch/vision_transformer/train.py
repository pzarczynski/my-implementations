import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from vit import VisionTransformer

BATCH_SIZE = 128
PATCH_SIZE = 8
NUM_CLASSES = 10
EPOCHS = 10


def load_dataset(dataset, batch_size):
    train_dataset = dataset("./data", download=True, transform=T.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    val_dataset = dataset("./data", download=True, train=False, transform=T.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def validate_model(model, loader, device, num_classes):
    model.eval()
    accuracy = Accuracy("multiclass", num_classes=num_classes).to(device)

    for inp, tgt in tqdm(loader):
        inp, tgt = inp.to(device), tgt.to(device)

        with torch.no_grad():
            preds = model(inp)

        accuracy.update(preds, tgt)

    return accuracy.compute()


def training_curve(loss, acc):
    fig, ax = plt.subplots()

    ax.plot(loss, label="loss")
    ax.plot(acc, label="accuracy")
    ax.set_xlabel("epochs")

    ax.legend()
    return fig, ax


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_dataset(torchvision.datasets.MNIST, BATCH_SIZE)

    model = VisionTransformer(patch_size=PATCH_SIZE, num_classes=NUM_CLASSES).to(device)

    optimizer = optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    epoch_loss, epoch_acc = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for inp, tgt in tqdm(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)

            preds = model(inp)

            loss = criterion(preds, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} train loss: {train_loss}")
        epoch_loss.append(train_loss)

        val_acc = validate_model(model, val_loader, device, NUM_CLASSES)
        print(f"Epoch {epoch+1} val accuracy: {val_acc}")
        epoch_acc.append(val_acc.cpu())

    fig, ax = training_curve(epoch_loss, epoch_acc)
    fig.savefig("training_curve.png")
