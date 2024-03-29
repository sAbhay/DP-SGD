from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp


def clipped_grad(params, l2_norm_clip, single_example_batch, loss):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads), total_grad_norm


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size, loss):
    """Return differentially private gradients for params, evaluated on batch."""
    # logger.info("Batch shape: {}".format(batch[0].shape, batch[1].shape))
    clipped_grads, total_grad_norm = vmap(clipped_grad, (None, None, 0, None))(params, l2_norm_clip, batch, loss)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), total_grad_norm

