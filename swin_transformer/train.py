from functools import partial
import argparse

import chex
import jax
import optax
from augmentations import augment
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
from helpers import load_cifar10, shuffle, batchify, compute_accuracy, plot_accuracy
from sgdr_schedule import make_sgdr_schedule, total_epochs
from jax import lax
from jax import numpy as jnp
from jax import random
from models import SwinTransformer

# Mean and std of the CIFAR10 dataset
MEAN = jnp.array([0.4914, 0.4822, 0.4465])
STD = jnp.array([0.2470, 0.2435, 0.2616])


class TrainState(train_state.TrainState):
    dynamic_scale: DynamicScale = None


def train_step(
    keystate: tuple[chex.PRNGKey, TrainState], 
    batch: tuple[chex.Array, ...],
) -> tuple[tuple[chex.PRNGKey, TrainState], tuple[chex.Array, chex.Array]]:
    """Perform one training step and return updated state, key and metrics."""
    key, state = keystate
    key, aug_key, drop_key = random.split(key, 3)
    
    inputs, labels = augment(aug_key, batch)
    inputs = (inputs - MEAN) / STD
    labels = optax.smooth_labels(labels, alpha=0.1)
    
    def loss_fn(params):        
        logits = state.apply_fn({'params': params}, inputs, rngs=drop_key)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        accuracy = compute_accuracy(logits, labels)
        return loss, accuracy
    
    if state.dynamic_scale:
        grad_fn = state.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, (loss, accuracy), grads = grad_fn(state.params)
        state = state.replace(dynamic_scale=dynamic_scale)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)
        
    state = state.apply_gradients(grads=grads)
        
    if state.dynamic_scale:
        select_fn = partial(jnp.where, is_fin)
        state = state.replace(
            opt_state=jax.tree_util.tree_map(select_fn, state.opt_state, state.opt_state),
            params=jax.tree_util.tree_map(select_fn, state.params, state.params),
        )
    return (key, state), (loss, accuracy) 


def val_step(
    state: TrainState, 
    batch: tuple[chex.Array, ...]
) -> tuple[TrainState, chex.Array]:
    """Perform one validation step and return state and metrics."""
    inputs, labels = batch
    inputs = (inputs - MEAN) / STD
    
    logits = state.apply_fn({'params': state.params}, inputs, eval=True)
    accuracy = compute_accuracy(logits, labels)
    return state, accuracy


def one_epoch(
    state: TrainState, 
    epoch: int, 
    key: chex.PRNGKey,
    train_dataset: tuple[chex.Array, chex.Array],
    val_dataset: tuple[chex.Array, chex.Array],
    batch_size: int
) -> tuple[TrainState, chex.Array]:
    shuffle_key, train_key = random.split(random.fold_in(key, epoch))
    train_dataset = shuffle(shuffle_key, *train_dataset)
    train_dataset = batchify(*train_dataset, batch_size=batch_size)
    
    (_, state), (loss, train_acc) = lax.scan(train_step, (train_key, state), train_dataset)
    _, val_acc = lax.scan(val_step, state, val_dataset)
    
    metrics = jnp.array([jnp.mean(x) for x in (loss, train_acc, val_acc)])
    
    jax.debug.print(
        "Epoch {}: train_loss \t{:.3f}; train_acc \t{:.2%}; val_acc \t{:.2%}",
        epoch + 1, *metrics
    )
    
    return state, metrics   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jax CIFAR10 Training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--t_0', type=float, default=10)
    parser.add_argument('--t_mul', type=float, default=1.2)
    parser.add_argument('--cycles', type=int, default=14)
    parser.add_argument('--t_warmup', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    
    args = parser.parse_args()
    
    key = random.key(args.seed)
    (x_train, y_train), (x_test, y_test) = load_cifar10(seed=args.seed)
    x_test, y_test = batchify(x_test, y_test, batch_size=args.bs)
    
    model = SwinTransformer()
    params = model.init(key, x_test[0])['params']
    print(model.tabulate(key, x_test[0], depth=2))
    
    n_batches = x_train.shape[0] // args.bs
    # schedule = make_sgdr_schedule(
    #     peak_value=args.lr, num_cycles=args.cycles, t_0=args.t_0,
    #     t_mul=args.t_mul, t_warmup=args.t_warmup, steps_per_epoch=n_batches
    # )
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=args.lr, warmup_steps=args.t_warmup * n_batches,
        decay_steps=args.epochs * n_batches, end_value=1e-6
    )
    tx = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.lion(schedule, weight_decay=args.weight_decay)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    epochs = args.epochs or total_epochs(args.t_0, args.t_mul, args.cycles, args.t_warmup)
    
    state, metrics = lax.scan(
        partial(
            one_epoch, key=key, train_dataset=(x_train, y_train), 
            val_dataset=(x_test, y_test), batch_size=args.bs
        ), 
        state, jnp.arange(epochs)
    )
    
    loss, train_acc, val_acc = metrics.transpose(1, 0)
    fig = plot_accuracy(train_acc, val_acc)
    fig.savefig('accuracy.png')
    