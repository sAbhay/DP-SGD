import jax.numpy as jnp
from jax import device_put_sharded


def shard_dataset(images, labels, devices):
    """Split a dataset into num_shards shards."""
    images = jnp.split(images, len(devices))
    labels = jnp.split(labels, len(devices))
    images = device_put_sharded(images, devices)
    labels = device_put_sharded(labels, devices)
    return images, labels
