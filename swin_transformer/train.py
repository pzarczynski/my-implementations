import jax
import chex
import optax

from jax import lax, random, numpy as jnp
from flax.training.train_state import TrainState

from models import SwinTransformer, SwinTransformerConfig
from helpers import *


def create_train_state(key: chex.PRNGKey, steps_per_epoch: int) -> tuple[TrainState, chex.PRNGKey]:
    """Create a TrainState from a newly initialized model and return an updated key."""
    key, init_key, dropout_key = random.split(key, 3)
    
    model = SwinTransformer()
    dummy_input = jnp.ones((64, 28, 28, 1))
    vars = model.init({'params': init_key, 'dropout': dropout_key}, dummy_input)
    
    cosine_kw = [
        dict(
            init_value=1e-6, 
            peak_value=1e-3,
            warmup_steps=steps_per_epoch * a,
            decay_steps=steps_per_epoch * b
        ) for (a, b) in [[5, 30], [0, 60]]
    ]
    
    scheduler = optax.sgdr_schedule(cosine_kw)
    optimizer = optax.adamw(scheduler, weight_decay=1e-1)
    
    state = TrainState.create(
        apply_fn=model.apply, 
        params=vars['params'], 
        tx=optimizer
    )
    return state, key


@jax.jit
def train_step(
    statekey: tuple[TrainState, chex.PRNGKey], 
    batch: tuple[chex.Array, ...]
) -> tuple[tuple[TrainState, chex.PRNGKey], tuple[chex.Array]]:
    """Perform one training step and return updated state, key and metrics."""
    state, key = statekey
    inputs, labels = batch
    key, dropout_key = random.split(key)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs, rngs={"dropout": dropout_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == labels)
        return loss, accuracy
        
    (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return (state, key), (loss, accuracy) 


@jax.jit
def val_step(state: TrainState, batch: tuple[chex.Array, ...]):
    """Perform one validation step and return state and metrics."""
    inputs, labels = batch
    logits = state.apply_fn({'params': state.params}, inputs, eval=True)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)
    return state, accuracy
    

def train_loop(num_epochs: int, batch_size: int = 128):
    key = random.key(42)
    
    x_train, y_train, x_test, y_test = load_openml('mnist_784')
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    
    state, key = create_train_state(key, x_train.shape[0] // batch_size)
    
    val_dataset = batchify(x_test, y_test, batch_size=batch_size)
    
    for epoch in range(num_epochs):
        train_dataset, key = shuffle(x_train, y_train, key=key)
        train_dataset = batchify(*train_dataset, batch_size=batch_size)
        
        (state, key), (loss, train_acc) = lax.scan(train_step, (state, key), train_dataset)
        _, val_acc = lax.scan(val_step, state, val_dataset)
        
        loss, train_acc, val_acc = map(jnp.mean, [loss, train_acc, val_acc])
        
        print("epoch {}: train_loss\t{:.3f}; train_acc\t{:.2%}; val_acc\t{:.2%}"
              .format(epoch + 1, loss, train_acc, val_acc))

    return state


if __name__ == '__main__':
    train_loop(90)