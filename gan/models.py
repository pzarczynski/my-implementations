import jax
import flax.linen as nn
import chex

from jax import image, random
    

def upsample(x: chex.Array, scale: float = 2.) -> chex.Array:
    B, H, W, C = x.shape
    x = image.resize(x, (B, int(H*scale), int(W*scale), C), "nearest")
    return x


class Generator(nn.Module):
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, x: chex.Array, train: bool = True) -> chex.Array:
        x = nn.ConvTranspose(128, (4, 4), 4)(x) # 4, 4, 128
        x = nn.leaky_relu(x, 0.1)
        
        for _ in range(3):
            # x = nn.ConvTranspose(64, (4, 4), 2)(x)        
            x = upsample(x)
            x = nn.Conv(64, (3, 3))(x)
            x = nn.BatchNorm(momentum=0.8)(x, use_running_average=not train)
            x = nn.leaky_relu(x, 0.1)
    
        x = nn.Conv(3, (3, 3))(x) # 64, 64, 3
        x = nn.tanh(x)
        return x
    
    def generate(self, batch_size: int, train: bool = True):
        key = self.make_rng('latent')
        z = random.normal(key, (batch_size, 1, 1, self.latent_dim))
        x = self(z, train=train)
        return x
    

class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x: chex.Array, train: bool = True) -> chex.Array:
        x = nn.Conv(64, (3, 3))(x) # 32, 32, 64
        x = nn.leaky_relu(x, 0.1)
        
        for fan_out in [128, 256]:        
            x = nn.Conv(fan_out, (4, 4), 3)(x)
            x = nn.BatchNorm(momentum=0.8)(x, use_running_average=not train)
            x = nn.leaky_relu(x, 0.1)
        
        x # 4, 4, 128
        x = x.max((1, 2)) # 128
        x = nn.Dropout(0.1, deterministic=False)(x)
        
        x = nn.Dense(1)(x)
        x = x.reshape(-1)
        return x
