from jax import grad
from jax import jit
from jax import random
from jax import vmap, tree_map
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

from sys import path as syspath
syspath.append('../../')
from common import util, log
logger = log.get_logger('aug_momentum')


def clipped_grad(params, l2_norm_clip, single_example_batch, loss):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads), total_grad_norm


def clipped_grad_single_aug_params(params, l2_norm_clip, batch, loss):
    clipped_grads, total_grad_norm = vmap(clipped_grad, (None, None, 0, None))(params, l2_norm_clip, batch, loss)
    logger.info("Total grad norm shape: {}".format(total_grad_norm.shape))
    return clipped_grads, total_grad_norm


def private_grad(params, batch, rng, velocity, l2_norm_clip, noise_multiplier,
                 batch_size, loss, augmult, mult_radius):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads, total_grad_norm = vmap(clipped_grad, (None, None, 0, None))(params, l2_norm_clip, batch, loss)
    mults = random.uniform(rng, shape=(augmult-1,), minval=-1, maxval=1) * mult_radius

    # aug_params = generate_augmult_perturbed_params(params, velocity, mults, augmult-1)
    # clipped_grads, total_grad_norm = vmap(clipped_grad_single_aug_params, (0, None, None, None))(aug_params, l2_norm_clip, batch, loss)
    aug_clipped_grads = clipped_grads
    aug_total_norms = [total_grad_norm]
    for i in range(augmult-1):
        param = perturb_params_with_momentum(params, velocity, mults[i])
        clipped_grads, total_grad_norm = vmap(clipped_grad, (None, None, 0, None))(param, l2_norm_clip, batch, loss)
        aug_clipped_grads = tree_map(lambda g1, g2: g1 + g2, aug_clipped_grads, clipped_grads)
        aug_total_norms.append(total_grad_norm)
    total_aug_norms = jnp.vstack(aug_total_norms)
    # logger.info(f"Total aug norm shape: {total_aug_norms.shape}")
    total_grad_norm = jnp.mean(total_aug_norms, axis=1)
    # logger.info(f"Total grad norm shape: {total_grad_norm.shape}")
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), total_grad_norm, total_aug_norms


def generate_augmult_perturbed_params(params, velocity, mults, augmult):
    # augmult_params = util.tree_stack([params] * augmult)
    # aug_params = vmap(perturb_params_with_momentum, (0, None, 0))(augmult_params, velocity, mults)
    # return aug_params

    aug_params = []
    for i in range(augmult):
        aug_params.append(perturb_params_with_momentum(params, velocity, mults[i]))
    return aug_params




def perturb_params_with_momentum(params, velocity, mult):
    return tree_map(
        lambda p, v: p + mult * v,
        params,
        velocity,
    )
