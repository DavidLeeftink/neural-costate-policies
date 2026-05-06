import jax
import jax.numpy as jnp
from typing import NamedTuple

class RunningStats(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

def init_stats(shape) -> RunningStats:
    """Initializes running statistics with a small epsilon count to prevent div by zero."""
    return RunningStats(
        mean=jnp.zeros(shape),
        var=jnp.ones(shape),
        count=jnp.array(1e-4) 
    )

def update_stats(stats: RunningStats, batch: jnp.ndarray) -> RunningStats:
    """Updates the running mean and variance using Welford's online algorithm."""
    batch_count = batch.shape[0]
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)

    tot_count = stats.count + batch_count
    delta = batch_mean - stats.mean
    
    new_mean = stats.mean + delta * batch_count / tot_count
    
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / tot_count
    
    new_var = M2 / tot_count
    
    return RunningStats(mean=new_mean, var=new_var, count=tot_count)

def normalize_obs(stats: RunningStats, obs: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Standardizes observations (subtracts mean, divides by std)."""
    return (obs - stats.mean) / jnp.sqrt(stats.var + eps)