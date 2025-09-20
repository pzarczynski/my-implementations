import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import chex
import jax

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


def batchify(dataset: tuple[chex.Array], batch_size: int) -> tuple[chex.Array]:
    """Reshape the data into batches, discarding remainder."""
    n_batches = dataset[0].shape[0] // batch_size
    batched = tuple(map(
        lambda t: t[:n_batches*batch_size]
            .reshape(n_batches, batch_size, -1), 
        dataset
    ))
    return batched


def shuffle(dataset: tuple[chex.Array], key: chex.PRNGKey) -> tuple[tuple[chex.Array], chex.PRNGKey]:
    """Permute the data and return the updated key."""
    key, shuffle_key = jax.random.split(key)
    idx = jax.random.permutation(shuffle_key, dataset[0].shape[0])
    shuffled = tuple(map(lambda t: t[idx], dataset))
    return shuffled, key