# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

"""Defines train and evaluation functions that compute losses and metrics."""

from typing import Any, Dict, Tuple
import logging

import chex
import haiku as hk
import jax.numpy as jnp
from jax_privacy.src.training import metrics as metrics_module
import optax
import jax
from functools import reduce

Model = hk.TransformedWithState


class MultiClassForwardFn:
  """Defines forward passes for multi-class classification."""

  def __init__(self, net: Model):
    """Initialization function.

    Args:
      net: haiku model to use for the forward pass.
    """
    self._net = net

  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: chex.ArrayTree,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
    """Initializes the model.

    Args:
      rng_key: random number generation key used for the random initialization.
      inputs: model inputs (of format `{'images': images, 'labels': labels}`),
        to infer shapes of the model parameters. Images are expected to be of
        shape [NKHWC] (K is augmult).
    Returns:
      Initialized model parameters and state.
    """
    images = inputs['images']
    # Images has shape [NKHWC] (K is augmult).
    init_model = self._net.init(rng_key, images[:, 0], is_training=True)

    logging.info("Number of parameters: %d", hk.data_structures.tree_size(init_model[0]))
    return init_model

  def augmult_forward(
      self,
      params: chex.ArrayTree,
      inputs: chex.ArrayTree,
      network_state: chex.ArrayTree,
      rng: chex.PRNGKey,
      frozen_params: chex.ArrayTree,
  ) -> Tuple[chex.Array, Any]:
    """Forward pass per example (training time).

    Args:
      params: model parameters that should get updated during training.
      inputs: model inputs (of format `{'images': images, 'labels': labels}`),
        where the labels are one-hot encoded. `images` is expected to be of
        shape [NKHWC] (K is augmult), and `labels` of shape [NKO].
      network_state: model state.
      rng: random number generation key.
      frozen_params: model parameters that should remain frozen. These will be
        merged with `params` before the forward pass, but are specified
        separately to make it easy to compute gradients w.r.t. the first
        argument `params` only.
    Returns:
      loss: loss function computed per-example on the mini-batch (averaged over
        the K augmentations).
      auxiliary information, including the new model state, metrics computed
        on the current mini-batch and the current loss value per-example.
    """
    images, labels = inputs['images'], inputs['labels']

    # `images` has shape [NKHWC] (K is augmult), while model accepts [NHWC], so
    # we use a single larger batch dimension.
    reshaped_images = images.reshape((-1,) + images.shape[2:])
    reshaped_labels = labels.reshape((-1,) + labels.shape[2:])

    all_params = hk.data_structures.merge(params, frozen_params)

    logits, network_state = self._net.apply(
        all_params, network_state, rng, reshaped_images, is_training=True)
    loss = self._loss(logits, reshaped_labels)

    # We reshape back to [NK] and average across augmentations.
    loss = loss.reshape(images.shape[:2])
    loss = jnp.mean(loss, axis=1)

    # Accuracy computation is performed with the first augmentation.
    logits = logits.reshape(images.shape[:2] + logits.shape[1:])
    selected_logits = logits[:, 0, :]
    labels = jnp.mean(labels, axis=1)

    metrics = self._train_metrics(selected_logits, labels)
    return jnp.mean(loss), (network_state, metrics, loss)

  def grad_norm(self, params, inputs, network_state, rng):
    inputs_expanded = jax.tree_map(
      lambda x: jnp.expand_dims(x, axis=0),
      inputs,
    )
    grad = jax.grad(self.grad_forward)(params, inputs_expanded, network_state, rng)
    norm = optax.global_norm(grad)
    return norm

  def forward(
      self,
      params: chex.ArrayTree,
      inputs: chex.ArrayTree,
      network_state: chex.ArrayTree,
      rng: chex.PRNGKey,
  ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Forward pass per example (evaluation time).

    Args:
      params: model parameters that should get updated during training.
      inputs: model inputs (of format `{'images': images, 'labels': labels}`),
        where the labels are one-hot encoded. `images` is expected to be of
        shape [NHWC], and `labels` of shape [NO].
      network_state: model state.
      rng: random number generation key.
    Returns:
      logits: logits computed per-example on the mini-batch.
      metrics: metrics computed on the current mini-batch.
    """
    logits, unused_network_state = self._net.apply(
      params, network_state, rng, inputs['images'])
    loss = jnp.mean(self._loss(logits, inputs['labels']))

    # grad_norms = jax.vmap(self.grad_norm, (None, None, None, 0, 0))(params, network_state, rng,
    #                                                                 jnp.expand_dims(inputs['images'], 1),
    #                                                                 jnp.expand_dims(inputs['labels'], 1))
    # accuracies = jnp.argmax(logits, axis=-1) == jnp.argmax(inputs['labels'], axis=-1)

    metrics = {'loss': loss, **self._eval_metrics(logits, inputs['labels'])}
    return logits, metrics

  def grad_forward(
      self,
      params: chex.ArrayTree,
      inputs: chex.ArrayTree,
      network_state: chex.ArrayTree,
      rng: chex.PRNGKey,
  ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Forward pass per example (evaluation time).

    Args:
      params: model parameters that should get updated during training.
      inputs: model inputs (of format `{'images': images, 'labels': labels}`),
        where the labels are one-hot encoded. `images` is expected to be of
        shape [NHWC], and `labels` of shape [NO].
      network_state: model state.
      rng: random number generation key.
    Returns:
      logits: logits computed per-example on the mini-batch.
      metrics: metrics computed on the current mini-batch.
    """
    logits, unused_network_state = self._net.apply(
      params, network_state, rng, inputs['images'])
    loss = jnp.mean(self._loss(logits, inputs['labels']))
    return loss

  def _loss(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Compute the loss per-example.

    Args:
      logits: logits vector of expected shape [...O].
      labels: one-hot encoded labels of expected shape [...O].
    Returns:
      Cross-entropy loss computed per-example on leading dimensions.
    """
    return optax.softmax_cross_entropy(logits, labels)

  def _train_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Dict[str, chex.Array]:
    return self._topk_accuracy_metrics(logits, labels)

  def _eval_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Dict[str, chex.Array]:
    return self._topk_accuracy_metrics(logits, labels)

  def _topk_accuracy_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Dict[str, chex.Array]:
    """Evaluates topk accuracy."""
    # NB: labels are one-hot encoded.
    acc1, acc5 = metrics_module.topk_accuracy(logits, labels, topk=(1, 5))
    metrics = {'acc1': 100 * acc1, 'acc5': 100 * acc5}
    return metrics
