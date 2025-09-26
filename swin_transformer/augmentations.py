from functools import partial

import chex
import dm_pix as pix
import jax
from flax.struct import dataclass
from jax import lax
from jax import numpy as jnp
from jax import random


@dataclass
class AugmentationConfig:
    mixup_prob: float = 0.2
    cutmix_prob: float = 0.0
    cutout_prob: float = 0.0
    mixcut_prob: float = 0.0
    
    # MixCut also uses these parameters
    mixup_alpha: float = 0.8
    cutmix_size: int = 12
    cutout_size: int = 12
    
    # RandAug parameters
    brightness_delta: float = 0.2
    contrast_factor: tuple[float, float] = (0.8, 1.2)
    saturation_factor: tuple[float, float] = (0.8, 1.2)
    hue_delta: float = 0.1
    max_angle: int = 10
    max_shift: int = 4


def no_augment(key, x, y, x1, y2):
    return key, x, y


@partial(jax.jit, static_argnums=(5,))
def mixup(
    key: chex.PRNGKey, 
    x: chex.Array, 
    y: chex.Array, 
    x1: chex.Array, 
    y1: chex.Array, 
    alpha: float,
) -> tuple[chex.Array, chex.Array]:
    key, beta_key = random.split(key)
    gamma = random.beta(beta_key, alpha, alpha)
    x = x * gamma + x1 * (1 - gamma)
    y = y * gamma + y1 * (1 - gamma)
    return key, x, y


@partial(jax.jit, static_argnums=(5, 6))
def cutmix(
    key: chex.PRNGKey, 
    x: chex.Array, 
    y: chex.Array, 
    x1: chex.Array, 
    y1: chex.Array, 
    size: int,
    cut_size: int,
) -> tuple[chex.Array, chex.Array]:
    gamma = (cut_size / size) ** 2
    key, idx_key = random.split(key)
    idx = random.randint(idx_key, (2,), minval=0, maxval=size-cut_size)
    cut = lax.dynamic_slice(x1, (*idx, 0), (cut_size, cut_size, 3))
    x = lax.dynamic_update_slice(x, cut, (*idx, 0))
    y = y * (1 - gamma) + y1 * gamma
    return key, x, y


@partial(jax.jit, static_argnames=('size', 'cut_size'))
def cutout(
    key: chex.PRNGKey, 
    x: chex.Array, 
    y: chex.Array, 
    x1: chex.Array = None, 
    y1: chex.Array = None, 
    *, 
    size: int,
    cut_size: int,
) -> tuple[chex.Array, chex.Array]:
    key, idx_key = random.split(key)
    idx = random.randint(idx_key, (2,), minval=0, maxval=size-cut_size)
    out = jnp.zeros((cut_size, cut_size, 3))
    x = lax.dynamic_update_slice(x, out, (*idx, 0))
    # y += (1 / y.shape[-1] - y) * (cut_size / size) ** 2
    return key, x, y


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def mixcut(key, x, y, x1, y1, size, mixup_alpha, cutmix_size, cutout_size):
    key, x, y = mixup(key, x, y, x1, y1, alpha=mixup_alpha)
    
    key, choice_key = random.split(key)
    key, x, y = lax.cond(
        0.5 < random.uniform(choice_key),
        no_augment, 
        partial(cutmix, size=size, cut_size=cutmix_size),
        key, x, y, x1, y1,
    )
    
    key, x, y = cutout(key, x, y, size=size, cut_size=cutout_size)
    return key, x, y


@partial(jax.jit, static_argnums=(1, 2, 3, 4), 
         static_argnames=('size', 'mixup_alpha', 'cutmix_size', 'cutout_size'))
def random_augmentation(
    key: chex.PRNGKey, 
    mixup_p: float,
    cutmix_p: float, 
    cutout_p: float,
    mixcut_p: float,
    *operands,  
    size: int, 
    mixup_alpha: float,
    cutmix_size: int,
    cutout_size: int,
) -> tuple[chex.Array, chex.Array]:
    no_aug_p = 1 - (mixup_p + cutmix_p + cutout_p + mixcut_p)
    probs = jnp.array([no_aug_p, mixup_p, cutmix_p, cutout_p, mixcut_p])
    key, idx_key = random.split(key)
    idx = random.categorical(idx_key, jnp.log(probs))
    branches = [
        no_augment,
        partial(mixup, alpha=mixup_alpha), 
        partial(cutmix, size=size, cut_size=cutmix_size),
        partial(cutout, size=size, cut_size=cutout_size),
        partial(mixcut, size=size, mixup_alpha=mixup_alpha, 
                cutmix_size=cutmix_size, cutout_size=cutout_size)
    ]
    return lax.switch(idx, branches, key, *operands)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def random_color(
    key: chex.PRNGKey, 
    x: chex.Array, 
    brightness_delta: float, 
    contrast_factor: tuple[float, float],
    saturation_factor: tuple[float, float],
    hue_delta: float,
) -> tuple[chex.PRNGKey, chex.Array]:
    key, bright_key, cont_key, sat_key, hue_key = random.split(key, 5)
    x = pix.random_brightness(bright_key, x, max_delta=brightness_delta)
    x = pix.random_contrast(cont_key, x, *contrast_factor)
    x = pix.random_saturation(sat_key, x, *saturation_factor)
    x = pix.random_hue(hue_key, x, max_delta=hue_delta)
    return key, x

    
@partial(jax.jit, static_argnums=(2, 3, 4))
def random_geometric(key, x, size, max_shift, max_angle):
    key, flip_key, rot_key, crop_key = random.split(key, 4)
    x = pix.random_flip_left_right(flip_key, x)
    
    angle = random.randint(rot_key, (), minval=-max_angle, maxval=max_angle+1)
    x = pix.rotate(x, angle=jnp.pi * angle / 180)    
    
    x = pix.pad_to_size(x, size+2*max_shift, size+2*max_shift)
    x = pix.random_crop(crop_key, x, (size, size, 3))
    return key, x


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def randaug(key, x, size, brightness_delta, contrast_factor, 
            saturation_factor, hue_delta, max_shift, max_angle):
    key, x = random_color(
        key, x, brightness_delta=brightness_delta, contrast_factor=contrast_factor, 
        saturation_factor=saturation_factor, hue_delta=hue_delta
    )
    key, x = random_geometric(key, x, size=size, max_shift=max_shift, max_angle=max_angle)
    return key, x


@partial(jax.jit, static_argnums=(3,))
def random_choice(key, xs, ys, batch_size: int):
    key, idx_key = random.split(key)
    idx = random.randint(idx_key, (), minval=0, maxval=batch_size)
    return key, xs[idx], ys[idx]
    

def augment(
    key: chex.PRNGKey, 
    batch: tuple[chex.Array, chex.Array],
    cfg: AugmentationConfig = AugmentationConfig(),
) -> chex.Array:   
    inputs, labels = batch
    B, S, *_ = inputs.shape
    
    randaug_fn = partial(
        randaug, size=S, brightness_delta=cfg.brightness_delta, 
        hue_delta=cfg.hue_delta, contrast_factor=cfg.contrast_factor, 
        saturation_factor=cfg.saturation_factor, 
        max_shift=cfg.max_shift, max_angle=cfg.max_angle
    )
        
    def augment_one(key, x, y):
        key, x = randaug_fn(key, x)
        
        key, x1, y1 = random_choice(key, inputs, labels, batch_size=B)
        key, x1 = randaug_fn(key, x)
        
        key, x, y = random_augmentation(
            key, cfg.mixup_prob, cfg.cutmix_prob, 
            cfg.cutout_prob, cfg.mixcut_prob,
            x, y, x1, y1, size=S, 
            mixup_alpha=cfg.mixup_alpha,
            cutmix_size=cfg.cutmix_size,
            cutout_size=cfg.cutout_size,
        )
        return x, y
        
    return jax.vmap(augment_one)(random.split(key, B), inputs, labels)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from helpers import load_cifar10
    
    x1, _, y1, _ = load_cifar10(take=0.001)
    x1, y1 = jnp.array(x1), jnp.array(y1)
    key, x_key, y_key = random.split(random.key(42), 3)
    
    def vec_to_str(x: chex.Array) -> str:
        return "(" + ', '.join((str(int(v)) if v in (0, 1) 
                                else f'{v:.2f}') for v in x) + ")"
    
    def show_images(xs, ys, s=(4, 4)):
        fig, axarr = plt.subplots(s[0], s[1], figsize=(3 * s[1], s[0]))
        for x, y, ax in zip(xs, ys, axarr.ravel()):
            ax.imshow(x)
            ax.axis('off')
            ax.set_title(vec_to_str(y))
        fig.tight_layout()
        plt.show()
    
    x2, y2 = augment(key, (x1, y1))
    x = jnp.stack([x1, x2], axis=1).reshape(-1, *x1.shape[1:])
    y = jnp.stack([y1, y2], axis=1).reshape(-1, *y1.shape[1:])
    show_images(x, y, s=(12, 2))
    