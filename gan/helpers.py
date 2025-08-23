import os
import numpy as np
import jax
import chex
import zipfile
import gdown
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from tqdm import tqdm

CELEBA_URL = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"


def load_celeba(
    root_dir: str = './tmp',
    shape: int = 32,
    test_size: float = 0,
    *,
    seed: int = 42
) -> tuple[chex.Array, ...]:
    os.makedirs(root_dir, exist_ok=True)
    archive_path = os.path.join(root_dir, "celeba.npz")
    
    if os.path.exists(archive_path):
        archive = np.load(archive_path)
        x = archive['x']
    else:
        zip_path = os.path.join(root_dir, "celeba.zip")
        if not os.path.exists(zip_path):
            gdown.download(CELEBA_URL, output=zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir) 
        
        images = []
        unzip_dir = os.path.join(root_dir, "img_align_celeba")
        for file in tqdm(os.listdir(unzip_dir)):
            img_path = os.path.join(unzip_dir, file)
            img = cv2.imread(img_path)[..., ::-1]
            img = cv2.resize(img, (shape, shape))
            images.append(img)
            
        x = np.stack(images, axis=0)
        x = x.astype(np.float32) / 127.5 - 1
        np.savez(archive_path, x=x)
        
    if test_size > 0:
        return train_test_split(x, test_size=test_size, random_state=seed)
    return x
    

def shuffle(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
    idx = jax.random.permutation(key, x.shape[0])
    return x[np.asarray(idx)]


def batchify(x: chex.Array, batch_size: int = 64):
    n_batches = x.shape[0] // batch_size
    x = x[:n_batches*batch_size]
    x = x.reshape(n_batches, batch_size, *x.shape[1:])
    return x


def plot_samples(samples, nrows=4, ncols=4):
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    for sample, ax in zip(samples, axarr.ravel()):
        ax.imshow(sample)
        ax.axis('off')
    plt.tight_layout()
    return fig


def plot_metrics(metrics: dict[str, list[float]]):
    fig = plt.figure(figsize=(8, 5))
    
    for k, v in metrics.items():
        x = np.arange(len(v))
        plt.plot(x, v, label=k)

    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig