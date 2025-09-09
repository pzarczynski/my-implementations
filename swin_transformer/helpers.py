import jax
import chex
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jax import lax, random, numpy as jnp, nn as jnn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from functools import partial
from matplotlib.ticker import MaxNLocator

sns.set_theme(context='paper', style='ticks')


def plot_metrics(
    metrics: list[chex.Array], name: str, ax=None,
    subtypes: list[str] = ["train", "validation"],
):
    metrics = np.array(metrics)
    x = np.arange(metrics.shape[1]) + 1    
    
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_ylabel(name.lower().capitalize()) 
    
    for y, subtype in zip(metrics, subtypes):
        label = f"{subtype} {name}".lower().capitalize()
        sns.lineplot(x=x, y=y, ax=ax, markers='o', label=label)
    
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine(ax.figure, right=False)
    return fig


def load_cifar10(test_size: float = 0.2, seed: int = 42, take: float = 1.0) -> tuple[chex.Array, ...]:
    bunch = fetch_openml('cifar_10_small', as_frame=False, parser='liac-arff')
    x = bunch.data.astype(jnp.float32) / 255
    take_idx = int(x.shape[0] * take)
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[:take_idx]
    y = bunch.target.astype(jnp.int64)
    y = jnn.one_hot(y, num_classes=10)[:take_idx]
    return train_test_split(x, y, test_size=test_size, random_state=seed)


@jax.jit
def shuffle(key: chex.PRNGKey, x: chex.Array, y: chex.Array
) -> tuple[chex.PRNGKey, tuple[chex.Array, chex.Array]]:
    idx = random.permutation(key, x.shape[0])
    return x[idx], y[idx]


@partial(jax.jit, static_argnames=['batch_size'])
def batchify(x: chex.Array, y: chex.Array, batch_size: int
) -> tuple[chex.Array, chex.Array]:
    n_batches = x.shape[0] // batch_size
    x = lax.dynamic_slice_in_dim(x, 0, n_batches*batch_size, axis=0)
    x = x.reshape(n_batches, batch_size, *x.shape[1:])
    y = lax.dynamic_slice_in_dim(y, 0, n_batches*batch_size, axis=0)
    y = y.reshape(n_batches, batch_size, *y.shape[1:])
    return x, y


@jax.jit
def compute_accuracy(logits: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1))