# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WideResNet implementation in JAX using Haiku."""

from typing import Any, Dict, Optional, Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from .activations import activations_dict
from .layers import WSConv2D


class _WideResNetBlock(hk.Module):
  """Block of a WideResNet."""

  def __init__(self, num_filters, stride=1, projection_shortcut=False,
               activation=jax.nn.relu, conv_fn=hk.Conv2D, groups=8, norm_args=None, name=None):
    super().__init__(name=name)
    num_bottleneck_layers = 1
    self._activation = activation
    if norm_args is None:
      norm_args = {
          'create_offset': False,
          'create_scale': True,
          'groups': groups,
      }
    self._bn_modules = []
    self._conv_modules = []
    for i in range(num_bottleneck_layers + 1):
      s = stride if i == 0 else 1
      self._bn_modules.append(hk.GroupNorm(
          name='groupnorm_{}'.format(i),
          **norm_args))
      self._conv_modules.append(conv_fn(
          output_channels=num_filters,
          padding='SAME',
          kernel_shape=(3, 3),
          stride=s,
          with_bias=False,
          name='conv_{}'.format(i)))  # pytype: disable=not-callable
    if projection_shortcut:
      self._shortcut = conv_fn(
          output_channels=num_filters,
          kernel_shape=(1, 1),
          stride=stride,
          with_bias=False,
          name='shortcut')  # pytype: disable=not-callable
    else:
      self._shortcut = None

  def __call__(self, inputs, **norm_kwargs):
    x = inputs
    orig_x = inputs
    for i, (bn, conv) in enumerate(zip(self._bn_modules, self._conv_modules)):
      x = bn(x, **norm_kwargs)
      x = self._activation(x)
      if self._shortcut is not None and i == 0:
        orig_x = x
      x = conv(x)
    if self._shortcut is not None:
      shortcut_x = self._shortcut(orig_x)
      x += shortcut_x
    else:
      x += orig_x
    return x


class WideResNet(hk.Module):
  """WideResNet designed for CIFAR-10."""

  def __init__(self,
               num_classes: int = 10,
               depth: int = 28,
               width: int = 10,
               activation: str = 'relu',
               conv_fn: Callable = hk.Conv2D,
               groups: int = 8,
               norm_args: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet, self).__init__(name=name)
    if (depth - 4) % 6 != 0:
      raise ValueError('depth should be 6n+4.')
    self._activation = getattr(jax.nn, activation)
    if norm_args is None:
      norm_args = {
          'create_offset': True,
          'create_scale': True,
          'groups': groups,
      }
    self._conv = conv_fn(
        output_channels=16,
        kernel_shape=(3, 3),
        stride=1,
        with_bias=False,
        name='init_conv')  # pytype: disable=not-callable
    self._bn = hk.GroupNorm(
        name='groupnorm',
        **norm_args)
    self._linear = hk.Linear(
        num_classes,
        w_init=jnp.zeros,
        name='logits')

    blocks_per_layer = (depth - 4) // 6
    filter_sizes = [width * n for n in [16, 32, 64]]
    self._blocks = []
    for layer_num, filter_size in enumerate(filter_sizes):
      blocks_of_layer = []
      for i in range(blocks_per_layer):
        stride = 2 if (layer_num != 0 and i == 0) else 1
        projection_shortcut = (i == 0)
        blocks_of_layer.append(_WideResNetBlock(
            num_filters=filter_size,
            stride=stride,
            projection_shortcut=projection_shortcut,
            activation=self._activation,
            norm_args=norm_args,
            name='resnet_lay_{}_block_{}'.format(layer_num, i)))
      self._blocks.append(blocks_of_layer)

  def __call__(self, inputs: chex.Array, **norm_kwargs) -> chex.Array:
    net = inputs
    net = self._conv(net)

    # Blocks.
    for blocks_of_layer in self._blocks:
      for block in blocks_of_layer:
        net = block(net, **norm_kwargs)
    net = self._bn(net, **norm_kwargs)
    net = self._activation(net)

    net = jnp.mean(net, axis=[1, 2])
    return self._linear(net)


def get_widresnet_fn(
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation: str = 'relu',
        weight_standardisation = True,
        groups: int = 8,
        norm_args: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
):
    def widresnet_model_fn(features, **_):
        conv_fn = hk.Conv2D
        if weight_standardisation:
            conv_fn = WSConv2D
        module = WideResNet(num_classes=num_classes, depth=depth, width=width, activation=activation, conv_fn=conv_fn, groups=groups, norm_args=norm_args, name=name)
        return module(features)
    return widresnet_model_fn
