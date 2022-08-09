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


def compute_epsilon(steps, num_examples, batch_size, noise_multiplier, target_delta=1e-5):
    if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    rdp_const = compute_rdp(q, noise_multiplier, steps, orders)
    eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
    return eps


def params_norm(params):
    nonempty_params, tree_def = tree_flatten(params)
    total_params_norm = jnp.linalg.norm(
        [jnp.linalg.norm(p.ravel()) for p in nonempty_params]
    )
    # logger.info("Params count:", sum([len(p.ravel()) for p in nonempty_params]))
    return total_params_norm