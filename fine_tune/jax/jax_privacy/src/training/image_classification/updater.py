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

"""The updater computes and applies the update.

Typical usage:
  # The updater requires a (haiku) init function, a forward function and a
  # batching instance.
  updater = updater.Updater(
        batching=batching,  # see `batching.py`
        train_init=train_init,  # init function of a haiku model
        forward=train_forward,  # see `forward.py`
        ...
  )

  ...

  # Initialize model and optimizer (pmapped).
  params, network_state, opt_state = updater.init(inputs, rng_key)

  # Apply update (pmapped).
  params, network_state, opt_state, stats = updater.update(
      params=params,
      network_state=network_state,
      opt_state=opt_state,
      global_step=global_step,
      inputs=inputs,
      rng=rng,
  )
"""

import functools
from typing import Any, Dict, Mapping, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
# from jax_privacy.src.training import grad_clipping
# from jax_privacy.src.training import optim
import optax


Model = hk.TransformedWithState
InitFn = Any
ForwardFn = Any


class Updater:
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      train_init: InitFn,
      forward: ForwardFn,
      noise_std_relative: Optional[chex.Numeric],
      clipping_norm: Optional[chex.Numeric],
      optimizer_name: str,
      optimizer_kwargs: Optional[Mapping[str, Any]],
      lr_init_value: chex.Numeric,
      lr_decay_schedule_name: Optional[str],
      lr_decay_schedule_kwargs: Optional[Mapping[str, Any]]
  ):
    """Initializes the updater.

    Args:
      batching: virtual batching that allows to use 'virtual' batches across
        devices and steps.
      train_init: haiku init function to initialize the model.
      forward: function that defines the loss function and metrics.
      noise_std_relative: standard deviation of the noise to add to the average
         of the clipped gradient to make it differentially private. It will be
         multiplied by `clipping_norm / batch_size` before the noise gets
         actually added.
      clipping_norm: clipping-norm for the per-example gradients (before
        averaging across the examples of the mini-batch).
      rescale_to_unit_norm: whether each clipped per-example gradient gets
        multiplied by `1 / clipping_norm`, so that the update is normalized.
        When enabled, the noise standard deviation gets adjusted accordingly.
      weight_decay: whether to apply weight-decay on the parameters of the model
        (mechanism not privatized since it is data-independent).
      train_only_layer: if set to None, train on all layers of the models. If
        specified as a string, train only layer whose name is an exact match
        of this string.
      optimizer_name: name of the optax optimizer to use.
      optimizer_kwargs: keyword arguments passed to optax when creating the
        optimizer (except for the learning-rate, which is handled in this
        class).
      lr_init_value: initial value for the learning-rate.
      lr_decay_schedule_name: if set to None, do not use any schedule.
        Otherwise, identifier of the optax schedule to use.
      lr_decay_schedule_kwargs: keyword arguments for the optax schedule being
        used.
      log_snr_global: whether to log the Signal-to-Noise Ratio (SNR) globally
        across layers, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
      log_snr_per_layer: whether to log the Signal-to-Noise Ratio (SNR) per
        layer, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
      log_grad_clipping: whether to log the proportion of per-example gradients
        that get clipped at each iteration.
      log_grad_alignment: whether to compute the gradient alignment: cosine
        distance between the differentially private gradients and the
        non-private gradients computed on the same data.
    """
    self._train_init = train_init
    self._forward = forward

    self._clipping_norm = clipping_norm

    self._optimizer_name = optimizer_name
    self._optimizer_kwargs = optimizer_kwargs
    self._lr_init_value = lr_init_value
    self._lr_decay_schedule_name = lr_decay_schedule_name
    self._lr_decay_schedule_kwargs = lr_decay_schedule_kwargs

    if clipping_norm in (float('inf'), None):
      # We can compute standard gradients.
      self._using_clipped_grads = False
      self.value_and_clipped_grad = functools.partial(
          jax.value_and_grad, has_aux=True)
    else:
      self._using_clipped_grads = True
      self.value_and_clipped_grad = functools.partial(
          grad_clipping.value_and_clipped_grad_vectorized,
          clipping_fn=grad_clipping.global_clipping(
              clipping_norm=clipping_norm,
              rescale_to_unit_norm=rescale_to_unit_norm,
          ),
          smoothing_hyperparams={'augmult': augmult, 'mult': mult},
      )

  def _regularization(self, params: chex.ArrayTree) -> chex.Array:
    l2_loss = optim.l2_loss(params)
    return self._weight_decay * l2_loss, l2_loss

  def _is_trainable(
      self,
      layer_name: str,
      unused_parameter_name: str,
      unused_parameter_value: chex.Array,
  ) -> bool:
    if self._train_only_layer:
      return layer_name == self._train_only_layer
    else:
      return True

  def init(
      self,
      *,
      inputs: chex.ArrayTree,
      rng_key: chex.PRNGKey,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    """Initialization function."""
    return self._pmapped_init(inputs, rng_key)

  @functools.partial(jax.pmap, static_broadcasted_argnums=0, axis_name='i')
  def _pmapped_init(
      self,
      inputs: chex.ArrayTree,
      rng_key: chex.PRNGKey,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    """Initialization function (to be pmapped)."""
    params, network_state = self._train_init(rng_key, inputs)

    trainable_params, unused_frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    opt_init, _ = optim.optimizer(
        optimizer_name=self._optimizer_name,
        optimizer_kwargs=self._optimizer_kwargs,
        learning_rate=0.0,
    )
    opt_state = opt_init(trainable_params)
    return params, network_state, opt_state

  def update(
      self,
      *,
      params: chex.ArrayTree,
      network_state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      global_step: chex.Array,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, Any]:
    """Perform the pmapped update."""
    # The function below is p-mapped, so arguments must be provided without name
    # and in the right order, hence why we define this method, which has to be
    # called with named arguments in order to avoid any mistake.
    return self._pmapped_update(
        params,
        network_state,
        opt_state,
        global_step,
        inputs,
        rng,
        utils.host_id_devices_for_rng(),
    )

  @functools.partial(jax.pmap, static_broadcasted_argnums=0, axis_name='i')
  def _pmapped_update(
      self,
      params: chex.ArrayTree,
      network_state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      global_step: chex.Array,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      host_id: Optional[chex.Array],
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, Any]:
    """Updates parameters."""
    # Note on rngs:
    # - rng is common across replicas thanks to config.random_train,
    # - therefore rng_common also common across replicas,
    # - rng_device is specialised per device (for independent randonmness).
    rng_tmp, rng_common = jax.random.split(rng)
    rng_device = utils.specialize_rng_host_device(
        rng_tmp, host_id, axis_name='i', mode='unique_host_unique_device')

    # Save the initial network state before it gets updated by a forward pass.
    initial_network_state = network_state

    # The update step is logged in the optimizer state (by optax.MultiSteps)
    #  under the name of 'gradient_step'.
    update_step = opt_state.gradient_step

    # Potentially split params between trainable parameters and frozen
    # parameters. Trainable parameters get updated, while frozen parameters do
    # not.
    params, frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    # Compute clipped-per-example gradients of the loss function (w.r.t. the
    # trainable parameters only).
    forward = functools.partial(self._forward, frozen_params=frozen_params)
    (loss, (network_state, metrics,
            loss_vector)), device_grads = self.value_and_clipped_grad(forward)(
                params, inputs, network_state, rng_device)

    if self._using_clipped_grads:
      device_grads, grad_norms_per_sample = device_grads
    else:
      grad_norms_per_sample = None

    # Synchronize metrics and gradients across devices.
    loss, metrics, avg_grads = jax.lax.pmean(
        (loss, metrics, device_grads), axis_name='i')
    loss_all = jax.lax.all_gather(loss_vector, axis_name='i')
    loss_vector = jnp.reshape(loss_all, [-1])

    # Compute the regularization and its corresponding gradients. Since those
    # are data-independent, there is no need to privatize / clip them.
    (reg, l2_loss), reg_grads = jax.value_and_grad(
        self._regularization, has_aux=True)(params)

    # Compute the noise scale based on `noise_std_relative`, the batch-size and
    # the clipping-norm. Here the noise is created by being added to a structure
    # of zeros mimicking the gradients structure.
    noise, std = optim.add_noise_to_grads(
        clipping_norm=self._clipping_norm,
        noise_std_relative=self._noise_std_relative,
        apply_every=self.batching.apply_update_every(global_step),
        total_batch_size=self.batching.batch_size(global_step),
        grads=jax.tree_map(jnp.zeros_like, avg_grads),
        rng_key=rng_common,
    )

    # Compute our 'final' gradients `grads`: these comprise the clipped
    # data-dependent gradients (`avg_grads`), the regularization gradients
    # (`reg_grads`) and the noise to be added to achieved differential privacy
    # (`noise`).
    grads = jax.tree_map(
        lambda *args: sum(args),
        avg_grads,
        reg_grads,
        noise,
    )

    # Compute the learning-rate according to its schedule. Note that the
    # schedule evolves with `update_step` rather than `global_step` since the
    # former accounts for the fact that gradient smay be accumulated over
    # multiple global steps.
    learning_rate = optim.learning_rate_schedule(
        update_step=update_step,
        init_value=self._lr_init_value,
        decay_schedule_name=self._lr_decay_schedule_name,
        decay_schedule_kwargs=self._lr_decay_schedule_kwargs,
    )

    # Create an optimizer that will only apply the update every
    # `k=self.batching.apply_update_every` steps, and accumulate gradients
    # in-between so that we can use a large 'virtual' batch-size.
    _, opt_apply = optim.optimizer(
        learning_rate=learning_rate,
        optimizer_name=self._optimizer_name,
        optimizer_kwargs=self._optimizer_kwargs
    )

    # Log all relevant statistics in a dictionary.
    scalars = dict(
        learning_rate=learning_rate,
        noise_std=std,
        train_loss=loss,
        train_loss_mean=jnp.mean(loss_vector),
        train_loss_min=jnp.min(loss_vector),
        train_loss_max=jnp.max(loss_vector),
        train_loss_std=jnp.std(loss_vector),
        train_loss_median=jnp.median(loss_vector),
        reg=reg,
        batch_size=self.batching.batch_size(global_step),
        data_seen=self.batching.data_seen(global_step),
        update_every=self.batching.apply_update_every(global_step),
        l2_loss=l2_loss,
        train_obj=(reg + loss),
        grads_norm=optax.global_norm(grads),
        params_norm=optax.global_norm(params),
        update_step=update_step,
    )

    scalars.update(metrics)

    # Perform the update on the model parameters (no-op if this step is meant to
    # accumulate gradients rather than performing the model update).
    updates, opt_state = opt_apply(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Merge the updated parameters with the parameters that are supposed to
    # remain frozen during training.
    new_params = hk.data_structures.merge(new_params, frozen_params)

    return new_params, network_state, opt_state, scalars
