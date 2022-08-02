from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import warnings

# https://github.com/tensorflow/privacy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent


def clipped_grad(params, l2_norm_clip, single_example_batch, loss, augmult):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    total_aug_norms = None
    if augmult > 0:
        def single_aug_grad(params, single_aug_batch):
            aug_grads = grad(loss)(params, single_aug_batch)
            nonempty_aug_grads, _ = tree_flatten(aug_grads)
            aug_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_aug_grads])
            return aug_grads, aug_grad_norm

        grads, total_aug_norms = vmap(single_aug_grad, (None, 0))(params, single_example_batch)
        # logger.info(f"Total aug norms: {total_aug_norms}")
        nonempty_grads, tree_def = tree_flatten(grads)
        # aug_norms = jnp.linalg.norm(jnp.hstack([jnp.linalg.norm(g, axis=0) for g in nonempty_grads]), axis=0).tolist()
        nonempty_grads = [g.mean(0) for g in nonempty_grads]
    else:
        grads = grad(loss)(params, single_example_batch)
        nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads), total_grad_norm, total_aug_norms


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size, loss, augmult):
    """Return differentially private gradients for params, evaluated on batch."""
    # logger.info("Batch shape: {}".format(batch[0].shape, batch[1].shape))
    clipped_grads, total_grad_norm, total_aug_norms = vmap(clipped_grad, (None, None, 0, None, None))(params, l2_norm_clip, batch, loss, augmult)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), total_grad_norm, total_aug_norms


def non_private_grad(params, batch, batch_size, loss):
    grads, total_grad_norm = vmap(grads_with_norm, (None, None, 0, None))(params, None, batch, loss)
    grads_flat, grads_treedef = tree_flatten(grads)
    aggregated_grads = [g.sum(0) / batch_size for g in grads_flat]
    return tree_unflatten(grads_treedef, aggregated_grads), total_grad_norm


def compute_epsilon(steps, num_examples, batch_size, noise_multiplier, target_delta=1e-5):
    if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    rdp_const = compute_rdp(q, noise_multiplier, steps, orders)
    eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
    return eps


def grads_with_norm(params, l2_norm_clip, single_example_batch, loss):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    return tree_unflatten(tree_def, nonempty_grads), total_grad_norm


def params_norm(params):
    nonempty_params, tree_def = tree_flatten(params)
    total_params_norm = jnp.linalg.norm(
        [jnp.linalg.norm(p.ravel()) for p in nonempty_params]
    )
    # logger.info("Params count:", sum([len(p.ravel()) for p in nonempty_params]))
    return total_params_norm