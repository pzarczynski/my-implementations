import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader

sns.set_theme(context='paper', style='ticks')


def plot_metrics(metrics):    
    x = np.arange(len(metrics['loss'])) + 1
    
    def one_plot(m, label: str, ax, c):
        y = np.array(m[label])
        sns.lineplot(x=x, y=y, ax=ax, color=c, marker='o')
        ax.set_ylabel(label.capitalize(), color=c)
        ax.tick_params(axis='y', colors=c)
        ax.set_yticks(np.array([y.min(), y.max()]).round(3))
    
    fig, ax = plt.subplots()
    one_plot(metrics, label='loss', ax=ax, c='blue')
    one_plot(metrics, label='accuracy', ax=ax.twinx(), c='red')
    
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    sns.despine(fig, right=False)
    return fig


def load_dataset(
    dataset: torchvision.datasets.VisionDataset, 
    batch_size: int,
    num_workers: int = os.cpu_count()
) -> tuple[DataLoader]:
    """Load the dataset and split it into train and val loaders."""
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_dataset = dataset("./tmp", download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataset = dataset("./tmp", download=True, train=False, transform=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, val_loader