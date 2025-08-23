import os
import jax
import chex
import optax

from jax import random, lax, numpy as jnp
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from helpers import *
from models import Generator, Discriminator


class GANTrainState(TrainState):
    batch_stats: FrozenDict


def create_train_states(key: chex.PRNGKey) -> tuple[GANTrainState, GANTrainState]:
    g_key, d_key = random.split(key)
    
    g = Generator()
    g_vars = g.init(g_key, jnp.ones((128, 1, 1, g.latent_dim)))
    g_tx = optax.adam(1e-4, b1=0.5)
    g_state = GANTrainState.create(
        apply_fn=g.apply, 
        params=g_vars['params'], 
        tx=g_tx,
        batch_stats=g_vars['batch_stats']
    )
    
    d = Discriminator()
    d_vars = d.init(d_key, jnp.ones((128, 32, 32, 3)))
    d_tx = optax.adam(3e-5, b1=0.5)
    d_state = GANTrainState.create(
        apply_fn=d.apply, 
        params=d_vars['params'], 
        tx=d_tx,
        batch_stats=d_vars['batch_stats']
    )
    return g_state, d_state


def train_step(
    stateskey: tuple[tuple[GANTrainState, GANTrainState], chex.PRNGKey], 
    batch: chex.Array,
) -> tuple[tuple[GANTrainState, chex.PRNGKey], tuple[chex.Array, chex.Array]]:
    B = batch.shape[0]
    (g_state, d_state), key = stateskey
    key, gen_key, dropout_key = random.split(key, 3)
    
    def d_loss_fn(d_params):
        g_vars = {'params': g_state.params,
                  'batch_stats': g_state.batch_stats}
        gen_images = g_state.apply_fn(
            g_vars, 
            batch_size=B, 
            train=False, 
            method='generate', 
            rngs={'latent': gen_key}
        )
        
        fake_images = lax.stop_gradient(gen_images)
        inputs = jnp.concat([batch, fake_images])
        labels = jnp.concat([0.9 * jnp.ones(B), jnp.zeros(B) + 0.1])
        
        d_vars = {'params': d_params,
                  'batch_stats': d_state.batch_stats}
        logits, d_stats = d_state.apply_fn(
            d_vars, 
            inputs, 
            mutable=['batch_stats'], 
            rngs={'dropout': dropout_key}
        )
        
        # loss = jnp.mean(jnp.square(logits - labels))
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        return loss, d_stats['batch_stats']
        
    (d_loss, d_stats), d_grads = jax.value_and_grad(d_loss_fn, has_aux=True)(d_state.params)
    d_state = d_state.apply_gradients(grads=d_grads).replace(batch_stats=d_stats)
    
    key, dropout_key = random.split(key)
        
    def g_loss_fn(g_params):
        g_vars = {'params': g_params,
                  'batch_stats': g_state.batch_stats}
        gen_images, g_stats = g_state.apply_fn(
            g_vars, 
            batch_size=B,  
            method='generate', 
            rngs={'latent': gen_key},
            mutable=['batch_stats']
        )
        
        d_vars = {'params': d_state.params,
                  'batch_stats': d_state.batch_stats}
        logits = d_state.apply_fn(
            d_vars, 
            gen_images, 
            train=False, 
            rngs={'dropout': dropout_key}
        )
        
        # loss = jnp.mean(jnp.square(logits - 1))
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, jnp.ones(B)))
        return loss, g_stats['batch_stats']
    
    (g_loss, g_stats), g_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g_state.params)
    g_state = g_state.apply_gradients(grads=g_grads).replace(batch_stats=g_stats)
    
    states = g_state, d_state
    return (states, key), (g_loss, d_loss)
    
    
def train_loop(epochs: int, batch_size: int = 128) -> None:
    images_dir = "img/"
    os.makedirs(images_dir, exist_ok=True)
    
    key = random.key(42)
    key, init_key = random.split(key)
    states = create_train_states(init_key)
    
    x = load_celeba()
    
    metrics = {'Generator loss': [], 'Discriminator loss': []}
    
    for epoch in range(epochs):
        key, shuffle_key = random.split(key)
        x = shuffle(x, key=shuffle_key)
        batches = batchify(x, batch_size=batch_size)
        
        (states, key), losses = lax.scan(train_step, (states, key), batches)
        g_loss, d_loss = map(jnp.mean, losses)
        
        print("epoch {}: train_g_loss\t{:.3f}; train_d_loss\t{:.3f}"
              .format(epoch + 1, g_loss, d_loss))
        
        metrics['Generator loss'].append(g_loss)
        metrics['Discriminator loss'].append(d_loss)
        
        g_state = states[0]
        key, gen_key = random.split(key)
        
        g_vars = {'params': g_state.params,
                  'batch_stats': g_state.batch_stats}
        samples = g_state.apply_fn(
            g_vars, 
            batch_size=100, 
            train=False,
            method='generate', 
            rngs={'latent': gen_key}
        )
        
        fig = plot_samples((samples + 1) / 2, 10, 10)
        fig.savefig(os.path.join(images_dir, f"samples_{epoch+1}"))
        plt.close(fig)
        
    plot_metrics(metrics).savefig("metrics")
        

if __name__ == '__main__':
    train_loop(100)