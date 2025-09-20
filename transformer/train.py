import jax
import jax.numpy as jnp
import optax
import chex

import logging
logging.basicConfig(level=logging.INFO)

from flax.training.train_state import TrainState
from omegaconf import DictConfig

from transformer import Transformer, TransformerConfig
from utils import plot_metrics, shuffle, batchify


def generate_dataset(
    size: int,
    max_len: int, 
    vocab_size: int,
    key: chex.PRNGKey
) -> tuple[tuple[chex.Array], chex.PRNGKey]:
    """Generate input and output sequences and return an updated key."""
    key, len_key, data_key = jax.random.split(key, 3)
    
    # Generate sequences of different lengths and mask them.
    seq_len = jax.random.randint(len_key, (size,), 1, max_len)
    mask = jnp.arange(max_len)[None, :] < seq_len[:, None]
    src = jax.random.randint(data_key, (size, max_len), 3, vocab_size)
    src *= mask
    
    # Shift the whole sequence by one then adjust 
    # the first element based on actual sequence length.
    last = src[jnp.arange(src.shape[0]), seq_len-1]
    tgt = jnp.roll(src, shift=1, axis=-1)
    
    # For each sequence set the first element 
    # to the last not masked element of `src` 
    # and the last element to <EOS> token (2).
    tgt = tgt.at[:, 0].set(last)
    tgt = tgt.at[jnp.arange(tgt.shape[0]), seq_len].set(2)
    # Prepend the data with <SOS> tokens (1).
    sos_tokens = jnp.ones((size, 1), dtype=jnp.int32)
    tgt = jnp.concat([sos_tokens, tgt], axis=-1)
    return (src, tgt), key


def create_train_state(
    encoder_cfg: TransformerConfig, 
    decoder_cfg: TransformerConfig, 
    lr: float, 
    key: chex.PRNGKey
) -> tuple[TrainState, chex.PRNGKey]:
    """Create a TrainState from a newly initialized model and return an updated key."""
    key, init_key, dropout_key = jax.random.split(key, 3)
    
    model = Transformer(encoder_cfg, decoder_cfg)
    dummy_seq = jnp.ones([1, encoder_cfg.max_len], dtype=jnp.int32)
    dummy_tgt = jnp.ones([1, decoder_cfg.max_len], dtype=jnp.int32)
    vars = model.init({'params': init_key, 'dropout': dropout_key}, dummy_seq, dummy_tgt)
    
    optimizer = optax.adam(learning_rate=lr)
    state = TrainState.create(
        apply_fn=model.apply, 
        params=vars["params"], 
        tx=optimizer
    )
    return state, key


@jax.jit
def compute_accuracy(preds: chex.Array, y: chex.Array) -> chex.Array:
    """Compute the accuracy between `preds` and `y` labels."""
    mask = y != 0
    accuracy = jnp.sum(mask * (preds == y)) / jnp.sum(mask)
    return accuracy


@jax.jit
def compute_loss(logits: chex.Array, y: chex.Array) -> chex.Array:
    """Compute the loss between `logits` and `y` labels."""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mask = y != 0
    loss = jnp.sum(mask * loss) / jnp.sum(mask)
    return loss


@jax.jit
def train_step(
    statekey: tuple[TrainState, chex.PRNGKey], 
    batch: tuple[chex.Array]
) -> tuple[tuple[TrainState, chex.PRNGKey], tuple[chex.Array]]:
    """Perform one training step and return updated state, key and metrics."""
    state, key = statekey
    src, tgt = batch
    key, dropout_key = jax.random.split(key)
    
    def loss_fn(params):
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = state.apply_fn(
            {"params": params}, 
            src, tgt_in, 
            rngs={"dropout": dropout_key}
        )
        loss = compute_loss(logits, tgt_out)
        preds = jnp.argmax(logits, axis=-1)
        acc = compute_accuracy(preds, tgt_out) 
        return loss, acc
        
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return (state, key), (loss, acc) 


@jax.jit
def val_step(state: TrainState, batch: tuple[chex.Array]):
    """Perform one validation step and return state and metrics."""
    src, tgt = batch
    
    preds = state.apply_fn(
        {"params": state.params}, src,
        method="generate", 
    )
    acc = compute_accuracy(preds, tgt[:, 1:])
    return state, acc
    

def train_loop(cfg: DictConfig):
    key = jax.random.PRNGKey(0)
    
    encoder_decoder_cfg = 2 * (TransformerConfig(vocab_size=cfg.vocab_size, max_len=cfg.max_len),)
    state, key = create_train_state(*encoder_decoder_cfg, lr=1e-3, key=key)
    
    size = cfg.n_batches * cfg.batch_size
    train_dataset, key = generate_dataset(size, cfg.max_len, cfg.vocab_size, key)
    val_dataset, key = generate_dataset(int(size * cfg.val_factor), cfg.max_len, cfg.vocab_size, key)
    batched_val_dataset = batchify(val_dataset, cfg.batch_size)
    
    metrics = dict(loss=[], accuracy=[])

    for epoch in range(cfg.num_epochs):
        train_dataset, key = shuffle(train_dataset, key)
        batched_train_dataset = batchify(train_dataset, cfg.batch_size)
        
        (state, key), (loss, train_acc) = jax.lax.scan(train_step, (state, key), batched_train_dataset)
        _, val_acc = jax.lax.scan(val_step, state, batched_val_dataset)
        
        loss, train_acc, val_acc = map(jnp.mean, [loss, train_acc, val_acc])
        
        logging.info("epoch {}: train_loss\t{:.3f}; train_acc\t{:.2%}; val_acc\t{:.2%}"
                     .format(epoch + 1, loss, train_acc, val_acc))
        
        metrics['loss'].append(loss)
        metrics['accuracy'].append(val_acc)

    return state, metrics


if __name__ == "__main__":
    cfg = DictConfig(dict(
        n_batches=50, 
        batch_size=128, 
        num_epochs=10,
        max_len=10,
        vocab_size=10,
        val_factor=0.1,
    ))
    final_state, metrics = train_loop(cfg)
        
    fig = plot_metrics(metrics)
    fig.savefig('curve.png')
