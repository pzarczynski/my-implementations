import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

import chex
import jax


def plot_metrics(metrics):
    assert 'loss' in metrics
    
    fig, ax = plt.subplots(1)
    x = np.arange(len(metrics['loss'])) + 1

    for label, y in metrics.items():
        ax.plot(x, y, marker='o', label=label)

    ax.set_xlabel('epoch')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax


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