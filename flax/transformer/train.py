import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.struct import dataclass
import flax.linen as nn

from functools import partial
from .transformer import TransformerEncoder, TransformerDecoder


SPC_TOK = [PAD_TOK, SOS_TOK, EOS_TOK] = range(3)

EPOCHS = 5
BATCH_SIZE = 128
TRAIN_NUM_BATCHES = 80
TRAIN_DATASET_SIZE = BATCH_SIZE * TRAIN_NUM_BATCHES
VAL_NUM_BATCHES = 4
VAL_DATASET_SIZE = BATCH_SIZE * VAL_NUM_BATCHES
SEQ_LEN = 16
VOCAB_SIZE = 128
EMBED_DIM = 128
NUM_HEADS = 8
MLP_DIM = 256
NUM_LAYERS = 6
DROPOUT_RATE = 0.1


class Transformer(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    vocab_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, src, tgt, dropout_eval=False):
        src_pad_mask = src != PAD_TOK
        tgt_pad_mask = tgt != PAD_TOK

        emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=1.0)
        )

        src, tgt = emb(src), emb(tgt)
    
        context = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
        )(
            src,
            pad_mask=src_pad_mask,
            dropout_eval=dropout_eval,
        )

        out = TransformerDecoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            vocab_size=self.vocab_size,
            dropout_rate=self.dropout_rate,
        )(
            tgt,
            context=context,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
            dropout_eval=dropout_eval,
        )
        return out


@jax.jit
def generate(state, src, key):
    batch_size = src.shape[0]
    tgt = jnp.zeros((batch_size, SEQ_LEN+1), dtype=jnp.int32)
    tgt = tgt.at[:, 0].set(SOS_TOK)

    def step_fn(step, carry):
        tgt, key = carry
        key, subkey = jax.random.split(key)
        
        logits = state.apply_fn(
            {'params': state.params},
            src,
            tgt,
            dropout_eval=True,
            rngs={'dropout': subkey},
        )
        
        next_token_logits = logits[:, step, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        tgt = tgt.at[:, step + 1].set(next_token)
        
        return tgt, key
        
    tgt, _ = jax.lax.fori_loop(0, SEQ_LEN, step_fn, (tgt, key))
    return tgt[:, 1:]


def generate_dataset(key, size, seq_len):
    src = jax.random.randint(key, (size, seq_len), len(SPC_TOK), VOCAB_SIZE - 1)
    tgt = jnp.concat([jnp.full((size, 1), SOS_TOK), jnp.flip(src, axis=-1)], axis=-1)
    return src, tgt


def batchify(src, tgt, batch_size):
    n_batches = src.shape[0] // batch_size
    src = src[: n_batches * batch_size].reshape(n_batches, batch_size, -1)
    tgt = tgt[: n_batches * batch_size].reshape(n_batches, batch_size, -1)
    return src, tgt


def shuffle(src, tgt, key):
    idx = jax.random.permutation(key, src.shape[0])
    return src[idx], tgt[idx]


def init_train_state(key, model, learning_rate):
    dummy_src = jnp.ones([1, SEQ_LEN], dtype=jnp.int32)
    dummy_tgt = jnp.ones([1, SEQ_LEN], dtype=jnp.int32)
    params = model.init(key, dummy_src, dummy_tgt)["params"]
    tx = optax.adamw(learning_rate, weight_decay=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


@jax.jit
def compute_accuracy(logits, labels):
    mask = labels != PAD_TOK
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.sum(mask * (preds == labels)) / jnp.sum(mask)
    return accuracy


@jax.jit
def compute_loss(params, batch, state, key):
    src, tgt = batch
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    logits = state.apply_fn({"params": params}, src, tgt_in, rngs={"dropout": key})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt_out)
    mask = tgt_out != PAD_TOK
    loss = jnp.sum(mask * loss) / jnp.sum(mask)
    acc = compute_accuracy(logits, tgt_out)
    return loss, acc


@jax.jit
def train_step(state, batch, key):
    key, subkey = jax.random.split(key)
    loss_fn = partial(compute_loss, batch=batch, state=state, key=subkey)
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc, key


@jax.jit
def train_epoch(state, src_batches, tgt_batches, key):
    def step_fn(carry, batch):
        state, key = carry
        state, loss, acc, key = train_step(state, batch, key)
        return (state, key), (loss, acc)

    (state, key), (losses, accuracies) = jax.lax.scan(
        step_fn, (state, key), (src_batches, tgt_batches)
    )
    return state, losses.mean(), accuracies.mean(), key


@jax.jit
def val_step(key, batch, state):
    key, subkey = jax.random.split(key)
    src, tgt = batch
    out = generate(state, src, subkey)
    tgt_out = tgt[:, 1:]
    mask = tgt_out != PAD_TOK
    acc = jnp.sum((out == tgt_out) * mask) / jnp.sum(mask)
    return key, acc


@jax.jit
def val_epoch(state, src_batches, tgt_batches, key):
    step_fn = partial(val_step, state=state)
    key, accuracies = jax.lax.scan(step_fn, key, (src_batches, tgt_batches))
    return accuracies.mean(), key
    

def train_loop(num_epochs, state, key):
    key, key_train, key_val = jax.random.split(key, 3)
    train_dataset = generate_dataset(key_train, TRAIN_DATASET_SIZE, SEQ_LEN)
    train_dataset = batchify(*train_dataset, BATCH_SIZE)

    val_dataset = generate_dataset(key_val, VAL_DATASET_SIZE, SEQ_LEN)
    val_dataset = batchify(*val_dataset, BATCH_SIZE)
    
    train_info = {'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        state, loss, _, key = train_epoch(state, *train_dataset, key)
        acc, key = val_epoch(state, *val_dataset, key)
        print("epoch {}: train_loss\t{:.3f}; val_acc\t{:.2%}".format(epoch + 1, loss, acc))
        train_info['loss'].append(loss)
        train_info['accuracy'].append(acc)
        
        key, subkey = jax.random.split(key)
        train_dataset = shuffle(*train_dataset, subkey)

    return state, train_info


if __name__ == "__main__":
    key = jax.random.PRNGKey(13)
    key, subkey = jax.random.split(key)

    model = Transformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE,
    )

    key, subkey = jax.random.split(key)
    state = init_train_state(subkey, model, learning_rate=1e-3)
    state, train_info = train_loop(EPOCHS, state=state, key=key)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def plot_curve(info):
        assert 'loss' in info
        
        fig, ax = plt.subplots(1)
        x = jnp.arange(len(info['loss'])) + 1

        for label, y in info.items():
            ax.plot(x, y, marker='o', label=label)

        ax.set_xlabel('epoch')
        ax.set_title('loss and accuracy over epochs')
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return fig, ax
        
    fig, ax = plot_curve(train_info)
    fig.savefig('transformer/curve.png')
