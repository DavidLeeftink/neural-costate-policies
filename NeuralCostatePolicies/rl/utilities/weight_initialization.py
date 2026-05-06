import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

def get_orthogonal_init(weight, key, scale=1.0):
    """Generates orthogonally initialized weights."""
    shape = weight.shape
    if len(shape) < 2:
        return weight
    
    flat_shape = (shape[0], jnp.prod(jnp.array(shape[1:])))
    a = jax.random.normal(key, flat_shape)
    u, _, v = jnp.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q

def init_weights(model: eqx.Module, key: jr.PRNGKey, scale: float = jnp.sqrt(2.0)) -> eqx.Module:
    """Applies orthogonal initialization to all Linear layers inside the given module."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=is_linear)
    
    new_leaves = []
    keys = jr.split(key, len(leaves))
    
    for i, leaf in enumerate(leaves):
        if is_linear(leaf):
            # Apply the requested scale
            new_weight = get_orthogonal_init(leaf.weight, keys[i], scale=scale)
            new_bias = jnp.zeros_like(leaf.bias) if leaf.bias is not None else None
            new_leaves.append(eqx.tree_at(lambda x: (x.weight, x.bias), leaf, (new_weight, new_bias)))
        else:
            new_leaves.append(leaf)
            
    return jax.tree_util.tree_unflatten(treedef, new_leaves)