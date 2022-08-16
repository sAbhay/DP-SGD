import jax.numpy as jnp
from jax import put_device_sharded, devices


def shard_dataset(images, labels, devices):
    """Split a dataset into num_shards shards."""
    images = jnp.split(images, len(devices))
    labels = jnp.split(labels, len(devices))
    images = put_device_sharded(images, devices)
    labels = put_device_sharded(labels, devices)
    return images, labels
