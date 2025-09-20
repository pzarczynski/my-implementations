import jax
import chex
import optax

from jax import lax, random, numpy as jnp
from flax.training.train_state import TrainState

from models import SwinTransformer
from helpers import (load_cifar10, shuffle, batchify, 
                     compute_accuracy, plot_metrics)


def lr_schedule(
    init_value: float,
    peak_value: float,
    num_cycles: int,
    T_0: int,
    T_mul: int,
    T_warmup: int,
    n_batches: int
):
    warmup_steps = T_warmup * n_batches
    decay_steps = T_0 * n_batches
    return optax.sgdr_schedule([
        dict(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps if cycle == 0 else 0,
            decay_steps=((warmup_steps if cycle == 0 else 0) +
                         (decay_steps * T_mul ** cycle)),
            end_value=init_value
        )
        for cycle in range(num_cycles)
    ])


def create_train_state(
    key: chex.PRNGKey, 
    dummy_input: chex.Array,
    n_batches: int
) -> tuple[TrainState, chex.PRNGKey]:
    """Create a TrainState from a newly initialized model and return an updated key."""
    model = SwinTransformer()
    params = model.init(key, dummy_input)['params']
    schedule = lr_schedule(init_value=1e-6, peak_value=2e-4,
                           num_cycles=10, T_0=10, T_mul=1.25, 
                           T_warmup=10, n_batches=n_batches)
    tx = optax.adamw(schedule, weight_decay=5e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


@jax.jit
def train_step(
    keystate: tuple[chex.PRNGKey, TrainState], 
    batch: tuple[chex.Array, ...]
) -> tuple[tuple[chex.PRNGKey, TrainState], tuple[chex.Array, chex.Array]]:
    """Perform one training step aax.debug.nd return updated state, key and metrics."""
    key, state = keystate
    inputs, labels = batch
    key, drop_key = random.split(key)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs, rngs=drop_key)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        accuracy = compute_accuracy(logits, labels)
        return loss, accuracy
        
    (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return (key, state), (loss, accuracy) 


@jax.jit
def val_step(
    state: TrainState, 
    batch: tuple[chex.Array, ...]
) -> tuple[TrainState, chex.Array]:
    """Perform one validation step and return state and metrics."""
    inputs, labels = batch
    logits = state.apply_fn({'params': state.params}, inputs, eval=True)
    accuracy = compute_accuracy(logits, labels)
    return state, accuracy


def train_loop(key: chex.PRNGKey, num_epochs: int, batch_size: int = 128):    
    x_train, x_test, y_train, y_test = load_cifar10(take=0.1)
    val_dataset = batchify(x_test, y_test, batch_size=batch_size)

    def one_epoch(state, epoch: int):
        shuffle_key, train_key = random.split(random.fold_in(key, epoch))
        
        train_dataset = shuffle(shuffle_key, x_train, y_train)
        train_dataset = batchify(*train_dataset, batch_size=batch_size)
        
        carry = train_key, state
        (_, state), (loss, train_acc) = lax.scan(train_step, carry, train_dataset)
        _, val_acc = lax.scan(val_step, state, val_dataset)
        
        metrics = jnp.array([jnp.mean(x) for x in (loss, train_acc, val_acc)])
        jax.debug.print("Epoch {}: train_loss \t{:.3f}; train_acc \t{:.2%}; val_acc \t{:.2%}", 
                        epoch, *metrics)
        
        return state, metrics
    
    state = create_train_state(
        key=key, 
        dummy_input=val_dataset[0][0], 
        n_batches=x_train.shape[0]//batch_size
    )
    state, metrics = lax.scan(one_epoch, state, jnp.arange(1, num_epochs+1))
    return state, metrics.transpose(1, 0)


if __name__ == '__main__':
    key = random.key(42)
    state, (_, *accuracies) = train_loop(key, num_epochs=2)
    
    fig = plot_metrics(accuracies, "accuracy")
    fig.tight_layout()
    fig.savefig('curve.png')