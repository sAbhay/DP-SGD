# As in jax_privacy/jax_privacy/src/training/image_classification/models/common.py

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import lax

class WSConv2D(hk.Conv2D):
  """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

  @hk.transparent
  def standardize_weight(self, weight, eps=1e-4):
    """Apply scaled WS with affine gain."""
    mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
    fan_in = np.prod(weight.shape[:-1])
    gain = hk.get_parameter('gain', shape=(weight.shape[-1],),
                            dtype=weight.dtype, init=jnp.ones)
    # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var).
    scale = lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
    shift = mean * scale
    return weight * scale - shift

  def __call__(self, inputs: chex.Array, eps: float = 1e-4) -> chex.Array:
    w_shape = self.kernel_shape + (
        inputs.shape[self.channel_index] // self.feature_group_count,
        self.output_channels)
    # Use fan-in scaled init, but WS is largely insensitive to this choice.
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'normal')
    w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
    weight = self.standardize_weight(w, eps)
    out = lax.conv_general_dilated(
        inputs, weight, window_strides=self.stride, padding=self.padding,
        lhs_dilation=self.lhs_dilation, rhs_dilation=self.kernel_dilation,
        dimension_numbers=self.dimension_numbers,
        feature_group_count=self.feature_group_count)
    # Always add bias.
    bias_shape = (self.output_channels,)
    bias = hk.get_parameter('bias', bias_shape, inputs.dtype, init=jnp.zeros)
    return out + bias