import jax

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )

def perturb_params_Gaussian(params, mult, rng_key):
    perturbation = tree_random_normal_like(rng_key, params)
    return jax.tree_util.tree_map(
        lambda p, v: p + mult * v,
        params,
        perturbation,
    )

def perturb_params_loop(params, augmult, dist, mult, key):
    keys = jax.random.split(key, num=augmult-1)
    if dist == 'Gaussian':
        aug_params = [params]
        for i in range(augmult-1):
            aug_params.append(perturb_params_Gaussian(params, mult, keys[i]))
        return aug_params, keys
    else:
        raise NotImplementedError("Supported perturbations: Gaussian")


def perturb_params_Gaussian_and_grad(params, mult, key, forward_fn, inputs_expanded, network_state):
    perturbed_params = perturb_params_Gaussian(params, mult, key)
    out, grad = jax.value_and_grad(forward_fn, has_aux=True)(perturbed_params, inputs_expanded, network_state, key)
    return grad
