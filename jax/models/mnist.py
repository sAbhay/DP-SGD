from .layers import WSConv2D
import haiku as hk
import jax

def get_mnist_model_fn(overparameterised=True, groups=8, ws=True):
    mult = 1
    if overparameterised:
        mult = 2

    def mnist_model_fn(features, **_):
        out = hk.Sequential([
            WSConv2D(16*mult, (8, 8), padding='SAME', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.GroupNorm(groups),
            WSConv2D(32*mult, (4, 4), padding='VALID', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.GroupNorm(groups),
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(10),
        ])(features)
        return out

    def mnist_model_fn_no_group(features, **_):
        out = hk.Sequential([
            WSConv2D(16*mult, (8, 8), padding='SAME', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            WSConv2D(32*mult, (4, 4), padding='VALID', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(10),
        ])(features)
        return out

    def mnist_model_fn_no_group_no_ws(features, **_):
        out = hk.Sequential([
            hk.Conv2D(16*mult, (8, 8), padding='SAME', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.Conv2D(32*mult, (4, 4), padding='VALID', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(10),
        ])(features)
        return out

    def mnist_model_fn_no_ws(features, **_):
        out = hk.Sequential([
            hk.Conv2D(16*mult, (8, 8), padding='SAME', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.GroupNorm(groups),
            hk.Conv2D(32*mult, (4, 4), padding='VALID', stride=(2, 2)),
            jax.nn.relu,
            hk.MaxPool(2, 1, padding='VALID'),  # matches stax
            hk.GroupNorm(groups),
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(10),
        ])(features)
        return out

    if groups == 0 and ws:
        return mnist_model_fn_no_group
    if groups == 0 and not ws:
        return mnist_model_fn_no_group_no_ws
    if groups > 0 and not ws:
        return mnist_model_fn_no_ws
    return mnist_model_fn
