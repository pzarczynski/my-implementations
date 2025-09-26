from functools import partial

import chex
import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

sns.set_theme(context='paper', style='ticks')


def load_cifar10(
    file: str = 'cifar10.hdf', 
    test_size: float = 1./6, 
    seed: int = 42, 
    take: float = 1.0
) -> tuple[tuple[chex.Array, chex.Array], tuple[chex.Array, chex.Array]]:
    with h5py.File(file, 'a') as f:
        try:
            x, y = f['x'][:], f['y'][:]
        except:
            bunch = fetch_openml('cifar_10', as_frame=False, parser='liac-arff')
            x = bunch.data.astype(jnp.float32) / 255
            x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y = bunch.target.astype(jnp.int64)
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
    
    take_idx = int(x.shape[0] * take)
    x = x[:take_idx]
    y = jnn.one_hot(y, num_classes=10)[:take_idx]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    return (x_train, y_train), (x_test, y_test)


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


def plot_accuracy(train: chex.Array, val: chex.Array):
    metrics = np.array([train, val])
    x = np.arange(metrics.shape[1]) + 1   
    fig, ax = plt.subplots()
    
    def plot_one(y, c, label, ha, va):
        sns.lineplot(x=x, y=y, ax=ax, color=c, label=label)
        max_idx = np.argmax(y)
        max_x, max_y = x[max_idx], y[max_idx]
        ax.scatter(max_x, max_y, color='black', s=5, zorder=5)
        ax.text(max_x, max_y, f'{max_y:.2%}', color='black', 
                fontsize=8, ha=ha, va=va)

    plot_one(train, c="red", label="Training accuracy", ha='left', va='top')
    plot_one(val, c="blue", label="Validation accuracy", ha='right', va='bottom')

    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine(fig)
    fig.tight_layout()
    return fig
