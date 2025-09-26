import numpy as np
import optax
import chex


def multipliers(x: float, num_cycles: int) -> chex.Array:
    return np.cumprod([1] + [x] * (num_cycles - 1))


def make_sgdr_schedule(
    peak_value: float,
    num_cycles: int,
    t_0: int,
    t_warmup: int,
    t_mul: int,
    steps_per_epoch: int,
    min_value: float = 1e-6,
) -> optax.Schedule:
    warmup_steps = t_warmup * steps_per_epoch
    decay_steps = np.floor(t_0 * multipliers(t_mul, num_cycles))
    # print(np.cumsum(decay_steps) + t_warmup)
    decay_steps *= steps_per_epoch
    decay_steps[0] += warmup_steps
    
    return optax.sgdr_schedule([
        dict(
            init_value=min_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps if cycle == 0 else 0,
            decay_steps=steps,
            end_value=min_value
        ) for cycle, steps in enumerate(decay_steps)
    ])


def total_epochs(t_0: int, t_mul: float, num_cycles: int, t_warmup: int) -> int:
    t = np.floor(t_0 * multipliers(t_mul, num_cycles))
    return int(np.sum(np.cumsum(t) + t_warmup))
    

def first_epoch(total_epochs: int, mul: float, num_cycles: int) -> int:
    return np.ceil(total_epochs / multipliers(mul, num_cycles).sum())