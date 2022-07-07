# Original example code: https://github.com/google/jax/blob/main/examples/differentially_private_sgd.py
# Additional reference: https://github.com/TheSalon/fast-dpsgd/blob/9584c9d4a6d061ca814005d02463537319b99faf/jaxdp.py

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX efficiently trains a differentially private conv net on MNIST.

This script contains a JAX implementation of Differentially Private Stochastic
Gradient Descent (https://arxiv.org/abs/1607.00133). DPSGD requires clipping
the per-example parameter gradients, which is non-trivial to implement
efficiently for convolutional neural networks.  The JAX XLA compiler shines in
this setting by optimizing the minibatch-vectorized computation for
convolutional architectures. Train time takes a few seconds per epoch on a
commodity GPU.

This code depends on tensorflow_privacy (https://github.com/tensorflow/privacy)
  Install instructions:
    $ pip install tensorflow
    $ git clone https://github.com/tensorflow/privacy
    $ cd privacy
    $ pip install .

The results match those in the reference TensorFlow baseline implementation:
  https://github.com/tensorflow/privacy/tree/main/tutorials

Example invocations:
  # this non-private baseline should get ~99% acc
  python -m examples.differentially_private_sgd \
    --dpsgd=False \
    --learning_rate=.1 \
    --epochs=20 \

   this private baseline should get ~95% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.3 \
   --l2_norm_clip=1.5 \
   --epochs=15 \
   --learning_rate=.25 \

  # this private baseline should get ~96.6% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.1 \
   --l2_norm_clip=1.0 \
   --epochs=60 \
   --learning_rate=.15 \

  # this private baseline should get ~97% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=0.7 \
   --l2_norm_clip=1.5 \
   --epochs=45 \
   --learning_rate=.25 \
"""

# TODO: clean up reshaping, move it out of predict and shape_as_image into dedicated functions

from common import log
logger = log.get_logger('experiment')

import itertools
import time
import warnings

from absl import app
from absl import flags

from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax import nn
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import haiku as hk

from data import datasets
import numpy.random as npr

# https://github.com/tensorflow/privacy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

import pickle

import models.mnist
from optim import sgdavg
from common import averaging

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('loss', 'cross-entropy', 'Loss function')
flags.DEFINE_boolean('overparameterised', True, 'Overparameterised for MNIST')
flags.DEFINE_integer('groups', None, 'Number of groups for GroupNorm, default None for no group normalisation')
flags.DEFINE_boolean('weight_standardisation', True, "Weight standardisation")
flags.DEFINE_boolean('parameter_averaging', True, "Parameter averaging")
flags.DEFINE_float('ema_coef', 0.999, "EMA parameter averaging coefficient")
flags.DEFINE_integer('ema_start_step', 0, "EMA start step")
flags.DEFINE_integer('polyak_start_step', 0, "Polyak start step")
flags.DEFINE_boolean('param_averaging', True, "Parameter averaging")
flags.DEFINE_string('image_shape', '28x28x1', "Augmult image shape")
flags.DEFINE_integer('augmult', 0, "Number of augmentation multipliers")
flags.DEFINE_boolean('random_flip', True, "Random flip augmentation")
flags.DEFINE_boolean('random_crop', True, "Random crop augmentation")

def main(_):
    logger.info("Running Experiment")

    if FLAGS.microbatches:
        raise NotImplementedError(
            'Microbatches < batch size not currently supported'
        )

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    logger.info(f"Train set shape: {train_images.shape}, {train_labels.shape}")
    if FLAGS.dpsgd and FLAGS.augmult > 0:
        start_time = time.time()
        image_size = [int(dim) for dim in FLAGS.image_shape.split("x")]
        train_images, train_labels = datasets.apply_augmult(train_images, train_labels,
                                                            image_size=image_size, augmult=FLAGS.augmult,
                                                            random_flip=FLAGS.random_flip, random_crop=FLAGS.random_crop)
        train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], -1))
        logger.info("Augmented train images in {:.2f} sec".format(time.time() - start_time))
    else:
        logger.warn("No data augmentation applied for vanilla SGD")
    logger.info(f"Train set shape: {train_images.shape}, {train_labels.shape}")
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
    num_batches = num_complete_batches + bool(leftover)
    key = random.PRNGKey(FLAGS.seed)

    def data_stream():
        rng = npr.RandomState(FLAGS.seed)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    def shape_as_image(images, labels, dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True):
        logger.info(f"Preshaped images shape: {images.shape}")
        target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
        if flatten_augmult:
            if augmult > 0:
                labels = jnp.reshape(labels, (FLAGS.batch_size * FLAGS.augmult, *labels.shape[2:]))
        elif augmult > 0:
            target_shape = (-1, augmult, 1, 28, 28, 1) if dummy_dim else (-1, augmult, 28, 28, 1)
        return jnp.reshape(images, target_shape), labels

    batches = data_stream()

    def average_params(params, add_params, t):
        return averaging.average_params(params, add_params, t,
                                        FLAGS.ema_coef, FLAGS.ema_start_step, FLAGS.polyak_start_step)
    opt_init, opt_update, get_params, set_params = sgdavg.sgd(FLAGS.learning_rate)

    rng = random.PRNGKey(42)
    model_fn = models.mnist.get_mnist_model_fn(FLAGS.overparameterised, FLAGS.groups)
    model = hk.transform(model_fn, apply_rng=True)

    init_batch = shape_as_image(*next(batches), dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True)[0]
    logger.info(f"Init batch shape: {init_batch.shape}")
    init_params = model.init(key, init_batch)
    def predict(params, inputs, augmult_reshape_input=True, augmult_reshape_preds=True):
        if augmult_reshape_input and FLAGS.augmult > 0:
            inputs = inputs.reshape(-1, *inputs.shape[2:])
        predictions = model.apply(params, None, inputs)
        if augmult_reshape_preds and FLAGS.augmult > 0:
            predictions = predictions.reshape(-1, FLAGS.augmult, *predictions.shape[1:])
        return predictions

    # jconfig.update('jax_platform_name', 'cpu')

    def ce_loss(params, batch):
      inputs, targets = batch
      logits = predict(params, inputs, augmult_reshape=True)
      logits = nn.log_softmax(logits, axis=-1)  # log normalize
      return -jnp.mean(jnp.mean(jnp.sum(logits * targets, axis=-1), axis=0))  # cross entropy loss

    def hinge_loss(params, batch):
        inputs, targets = batch
        if len(targets.shape) == 1:
            targets = targets.reshape(1, -1)
        target_class = jnp.argmax(targets, axis=-1)
        scores = predict(params, inputs, augmult_reshape=True)
        target_class_scores = jnp.choose(target_class, scores.T, mode='wrap')[:, jnp.newaxis]
        return jnp.mean(jnp.mean(jnp.sum(jnp.maximum(0, 1+scores-target_class_scores)-1, axis=-1), axis=0))


    if FLAGS.loss == 'cross-entropy':
        loss = ce_loss
    elif FLAGS.loss == 'hinge':
        loss = hinge_loss
    else:
        raise ValueError("Undefined loss")

    def accuracy(params, batch):
      inputs, targets = batch
      target_class = jnp.argmax(targets, axis=-1)
      logits = predict(params, inputs, augmult_reshape=False)
      predicted_class = jnp.argmax(logits, axis=-1)
      # logits_list = logits.tolist()
      # print(logits_list[0])
      return jnp.mean(predicted_class == target_class), predicted_class == target_class, logits


    def clipped_grad(params, l2_norm_clip, single_example_batch):
      """Evaluate gradient for a single-example batch and clip its grad norm."""
      logger.info("Single example batch: {}".format(single_example_batch[0].shape, single_example_batch[1].shape))
      if FLAGS.augmult > 0:
          grads = vmap(grad(loss), (None, 0))(params, single_example_batch, augmult_reshape_input=False, augmult_reshape_preds=False)
          nonempty_grads, tree_def = tree_flatten(grads)
          logger.info("Number of grads: {}".format(len(nonempty_grads), nonempty_grads))
          nonempty_grads = [g.mean(0) for g in grads]
      else:
          grads = grad(loss)(params, single_example_batch)
          nonempty_grads, tree_def = tree_flatten(grads)
      total_grad_norm = jnp.linalg.norm(
          [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
      divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
      normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
      return tree_unflatten(tree_def, normalized_nonempty_grads), total_grad_norm


    def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                     batch_size):
      """Return differentially private gradients for params, evaluated on batch."""
      logger.info("Batch shape: {}".format(batch[0].shape, batch[1].shape))
      clipped_grads, total_grad_norm = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
      clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
      aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
      rngs = random.split(rng, len(aggregated_clipped_grads))
      noised_aggregated_clipped_grads = [
          g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
          for r, g in zip(rngs, aggregated_clipped_grads)]
      normalized_noised_aggregated_clipped_grads = [
          g / batch_size for g in noised_aggregated_clipped_grads]
      return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), total_grad_norm


    def non_private_grad(params, batch, batch_size):
        grads, total_grad_norm = vmap(grads_with_norm, (None, None, 0))(params, None, batch)
        grads_flat, grads_treedef = tree_flatten(grads)
        aggregated_grads = [g.sum(0) / batch_size for g in grads_flat]
        return tree_unflatten(grads_treedef, aggregated_grads), total_grad_norm


    def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
      if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
      q = FLAGS.batch_size / float(num_examples)
      orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
      rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
      eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
      return eps


    def grads_with_norm(params, l2_norm_clip, single_example_batch):
      """Evaluate gradient for a single-example batch and clip its grad norm."""
      grads = grad(loss)(params, single_example_batch)
      nonempty_grads, tree_def = tree_flatten(grads)
      total_grad_norm = jnp.linalg.norm(
          [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
      return tree_unflatten(tree_def, nonempty_grads), total_grad_norm


    def params_norm(params):
        nonempty_params, tree_def = tree_flatten(params)
        total_params_norm = jnp.linalg.norm(
            [jnp.linalg.norm(p.ravel()) for p in nonempty_params]
        )
        logger.info("Params count:", sum([len(p.ravel()) for p in nonempty_params]))
        return total_params_norm

    @jit
    def update(_, i, opt_state, batch, add_params):
        params = get_params(opt_state)
        grads, total_grad_norm = non_private_grad(params, batch, FLAGS.batch_size)
        opt_state = opt_update(i, grads, opt_state)
        if FLAGS.param_averaging:
            params = get_params(opt_state)
            avg_params = average_params(params, add_params, i)
            opt_state = set_params(avg_params, params)
        return opt_state, total_grad_norm

    @jit
    def private_update(rng, i, opt_state, batch, add_params):
        params = get_params(opt_state)
        rng = random.fold_in(rng, i)  # get new key for new random numbers
        private_grads, total_grad_norm = private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                     FLAGS.noise_multiplier, FLAGS.batch_size)
        opt_state = opt_update(
            i, private_grads, opt_state)
        params = get_params(opt_state)
        avg_params = average_params(params, add_params, i)
        # logger.info(f"Average params: {avg_params}, \n Grads: {private_grads}")
        # logger.info("Optimization state: {}".format(opt_state))
        opt_state = set_params(avg_params, opt_state)
        return opt_state, total_grad_norm

    # _, init_params = init_random_params(key, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    grad_norms = []
    param_norms = []
    stats = []
    steps_per_epoch = 60000 // FLAGS.batch_size
    add_params = {'ema': get_params(opt_state), 'polyak': get_params(opt_state)}
    logger.info('Starting training...')
    for epoch in range(1, FLAGS.epochs + 1):
        start_time = time.time()
        epoch_grad_norms = []
        for _ in range(num_batches):
          next_batch = next(batches)
          if FLAGS.dpsgd:
            opt_state, total_grad_norm = private_update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params)
          else:
            opt_state, total_grad_norm = update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params)
          acc, correct, logits = accuracy(get_params(opt_state), shape_as_image(*next_batch))
          # print('Grad norm', len(total_grad_norm), 'Correct', len(correct))
          epoch_grad_norms += zip(total_grad_norm, correct, logits)
        param_norms.append(params_norm(get_params(opt_state)))

        grad_norms.append(epoch_grad_norms)

        # evaluate test accuracy
        params = get_params(opt_state)
        test_acc, _, _ = accuracy(params, shape_as_image(test_images, test_labels, augmult=FLAGS.augmult, flatten_augmult=True))
        test_loss = loss(params, shape_as_image(test_images, test_labels, augmult=FLAGS.augmult, flatten_augmult=True))
        logger.info('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            test_loss, 100 * test_acc))
        train_acc, _, _ = accuracy(params, shape_as_image(train_images, train_labels))
        train_loss = loss(params, shape_as_image(train_images, train_labels))
        logger.info('Train set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            train_loss, 100 * train_acc))

        # determine privacy loss so far
        if FLAGS.dpsgd:
            delta = 1e-5
            num_examples = 60000
            eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
            logger.info(
                'For delta={:.0e}, the current epsilon is: {:.2f}'.format(delta, eps))
        else:
            eps = None
            logger.info('Trained with vanilla non-private SGD optimizer')

        stats.append((train_loss, train_acc, test_loss, test_acc, eps))

        if epoch == FLAGS.epochs:
            if not FLAGS.dpsgd:
                hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging}"
            else:
                hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},nm={FLAGS.noise_multiplier},l2nc={FLAGS.l2_norm_clip},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging},aug={FLAGS.augmult},rf={FLAGS.random_flip},rc={FLAGS.random_crop}"
            with open(f'grad_norms_{hyperparams_string}.pkl', 'wb') as f:
                pickle.dump(grad_norms, f)
            with open(f'param_norms_{hyperparams_string}.pkl', 'wb') as f:
                pickle.dump(param_norms, f)
            with open(f'stats_{hyperparams_string}.pkl', 'wb') as f:
                pickle.dump(param_norms, f)

        epoch_time = time.time() - start_time
        logger.info('Epoch {} in {:0.2f} sec'.format(epoch, epoch_time))


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e