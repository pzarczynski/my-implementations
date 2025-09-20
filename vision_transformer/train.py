import torch
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

from vit import VisionTransformer
from utils import *
from omegaconf import DictConfig

torch.multiprocessing.set_start_method("spawn", force=True)


def train_epoch(
    model: VisionTransformer,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float]:
    """Perform training epoch, return mean loss and accuracy metrics."""
    model.train()
    accuracy = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=num_classes
    ).to(device)
    running_loss = torchmetrics.MeanMetric()

    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item())
        accuracy.update(logits, labels)

    return running_loss.compute(), accuracy.compute()
    

def val_epoch(
    model: VisionTransformer, 
    loader: DataLoader, 
    device: torch.device, 
    num_classes: int
) -> float:
    """Perform validation epoch, return the mean accuracy over steps."""
    model.eval()
    accuracy = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=num_classes
    ).to(device)

    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(inputs)

        accuracy.update(logits, labels)

    return accuracy.compute()


def training_loop(cfg: DictConfig, dataset) -> VisionTransformer:
    # Set the seed for reproducibility.
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    num_classes = len(dataset.classes)
    train_loader, val_loader = load_dataset(dataset, cfg.batch_size)

    model = VisionTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-5)
    
    metrics = dict(loss=[], accuracy=[])

    for epoch in range(cfg.num_epochs):
        loss, train_acc = train_epoch(
            model, optimizer, 
            loader=train_loader, 
            device=device, 
            num_classes=num_classes
        )
        val_acc = val_epoch(
            model, 
            loader=val_loader, 
            device=device, 
            num_classes=num_classes
        )
        
        metrics['loss'].append(loss.detach().cpu())
        metrics['accuracy'].append(val_acc.detach().cpu())
        
        logging.info("epoch {: 3}: loss\t{:.4f}; train_acc\t{:.2%}; val_acc\t{:.2%}; lr\t{:.2e}"
                     .format(epoch+1, loss, train_acc, val_acc, scheduler.get_last_lr()[-1]))
        
        scheduler.step()

    return model, metrics


if __name__ == '__main__':
    cfg = DictConfig(dict(
        num_epochs=10,
        batch_size=128,
        device="cuda",
        seed=42,
    ))
    
    model, metrics = training_loop(
        cfg, dataset=torchvision.datasets.MNIST)
    
    plot_metrics(metrics).savefig('curve')