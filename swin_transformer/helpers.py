import os
import jax
import jax.numpy as jnp
import chex

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_openml(
    dataset: str = 'mnist_784',
    root_dir: str = './tmp',
    test_size: float = 0.2,
    *,
    seed: int = 42
) -> tuple[chex.Array, ...]:
    os.makedirs(root_dir, exist_ok=True)
    path = os.path.join(root_dir, f'{dataset}.npz')
    try:
        archive = jnp.load(path)
        x, y = archive['x'], archive['y']
    except FileNotFoundError:
        bunch = fetch_openml(dataset, as_frame=False, parser='liac-arff')
        x = bunch.data.astype(jnp.float32) / 255
        y = bunch.target.astype(jnp.int64)
        jnp.savez(path, x=x, y=y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=test_size, 
        random_state=seed
    )
    return x_train, y_train, x_test, y_test


def shuffle(*xs: chex.Array, key: chex.PRNGKey) -> tuple[tuple[chex.Array, ...], chex.PRNGKey]:
    key, shuffle_key = jax.random.split(key)
    idx = jax.random.permutation(shuffle_key, xs[0].shape[0])
    return tuple(x[idx] for x in xs), key


def batchify(*xs: chex.Array, batch_size: int = 64) -> tuple[chex.Array, ...]:
    n_batches = xs[0].shape[0] // batch_size
    batched = tuple(map(
        lambda x: x[:n_batches*batch_size]
            .reshape(n_batches, batch_size, *x.shape[1:]), 
        xs
    ))
    return batched