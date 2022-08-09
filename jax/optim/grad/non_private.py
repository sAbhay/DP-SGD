from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp


def non_private_grad(params, batch, batch_size, loss):
    grads, total_grad_norm = vmap(grads_with_norm, (None, 0, None))(params, batch, loss)
    grads_flat, grads_treedef = tree_flatten(grads)
    aggregated_grads = [g.sum(0) / batch_size for g in grads_flat]
    return tree_unflatten(grads_treedef, aggregated_grads), total_grad_norm



def grads_with_norm(params, single_example_batch, loss):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    return tree_unflatten(tree_def, nonempty_grads), total_grad_norm