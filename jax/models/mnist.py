import haiku as hk
import jax

def mnist_model(features, **_):
    return hk.Sequential([
        hk.Conv2D(16, (8, 8), padding='SAME', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Conv2D(32, (4, 4), padding='VALID', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Flatten(),
        hk.Linear(32),
        jax.nn.relu,
        hk.Linear(10),
    ])(features)

def overparameterised_mnist_model(features, **_):
    return hk.Sequential([
        hk.Conv2D(32, (16, 16), padding='SAME', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Conv2D(64, (8, 8), padding='VALID', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Flatten(),
        hk.Linear(32),
        jax.nn.relu,
        hk.Linear(10),
    ])(features)

def get_mnist_model_fn(overparameterised: bool, groups=None):
    if overparameterised:
        model_fn = mnist_model
    else:
        model_fn = overparameterised_mnist_model

    if groups is not None:
        group_norm_fn = hk.GroupNorm(groups)
    else:
        group_norm_fn = lambda x: x

    def mnist_model_fn(features, **_):
        out = group_norm_fn(features)
        out = model_fn(features)
        return out

    return mnist_model_fn
