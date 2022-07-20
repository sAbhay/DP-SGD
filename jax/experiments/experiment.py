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

"""JAX efficiently trains a differentially private conv net on MNIST or CIFAR10.

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

# TODO: set up log debugging
# TODO: add virtual gradient accumulation

from sys import path as syspath

import jax.numpy

syspath.append('../')

from common import log
logger = log.get_logger('experiment')

import itertools
import time
import warnings

import psutil
import nvidia_smi

from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax import nn
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
import haiku as hk

import numpy.random as npr

# https://github.com/tensorflow/privacy
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

import pickle

from data import datasets
import models.models as models
from optim import sgdavg
from common import averaging

from absl import app
from absl import flags
from os import path as ospath

from analysis import make_plots
from image_concat import make_single_plot

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

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
# flags.DEFINE_string('image_shape', '28x28x1', "Augmult image shape")
flags.DEFINE_integer('augmult', 0, "Number of augmentation multipliers")
flags.DEFINE_boolean('random_flip', True, "Random flip augmentation")
flags.DEFINE_boolean('random_crop', True, "Random crop augmentation")
flags.DEFINE_string('norm_dir', 'norms', "Experiment data save directory")
flags.DEFINE_string('plot_dir', 'plots', "Experiment plots save directory")
flags.DEFINE_boolean('train', True, "Train")
flags.DEFINE_string('hyperparams_string', None, "Hyperparam string if not training")
flags.DEFINE_string('dataset', "mnist", "Dataset: mnist or cifar10")
flags.DEFINE_integer('augmult_batch_size', None, "Augmult batch size")

def log_memory_usage():
    logger.info(f"RAM usage: {psutil.virtual_memory()}")
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    logger.info(f'GPU usage: {100 * (mem_res.used / mem_res.total):.3f}%')

def experiment():
    logger.info("Running Experiment")
    log_memory_usage()

    if FLAGS.microbatches:
        raise NotImplementedError(
            'Microbatches < batch size not currently supported'
        )

    train_images, train_labels, test_images, test_labels = datasets.data(name=FLAGS.dataset)
    logger.info(f"Train set shape: {train_images.shape}, {train_labels.shape}")
    log_memory_usage()
    if FLAGS.dpsgd and FLAGS.augmult > 0:
        start_time = time.time()
        image_size = datasets.IMAGE_SHAPE[FLAGS.dataset]
        aug_train_images, aug_train_labels = datasets.apply_augmult(train_images, train_labels,
                                                            image_size=image_size, augmult=FLAGS.augmult,
                                                            random_flip=FLAGS.random_flip, random_crop=FLAGS.random_crop,
                                                            batch_size=FLAGS.augmult_batch_size)
        logger.info(f"Augmented train set shape: {aug_train_images.shape}, {aug_train_labels.shape}")
        logger.info("Augmented train images in {:.2f} sec".format(time.time() - start_time))
        log_memory_usage()
    else:
        logger.warn("No data augmentation applied for vanilla SGD")
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
    num_batches = num_complete_batches + bool(leftover)
    key = random.PRNGKey(FLAGS.seed)

    def data_stream(train_images, train_labels):
        rng = npr.RandomState(FLAGS.seed)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    def shape_as_image(images, labels, dataset=FLAGS.dataset, dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True):
        # logger.info(f"Preshaped images shape: {images.shape}")
        image_shape = datasets.IMAGE_SHAPE[dataset]
        target_shape = (-1, 1, *image_shape) if dummy_dim else (-1, *image_shape)
        if flatten_augmult:
            if augmult > 0:
                # logger.info(f"Preshaped labels shape: {labels.shape}")
                labels = jnp.reshape(labels, (-1, *labels.shape[2:]))
        elif augmult > 0:
            target_shape = (-1, augmult, 1, *image_shape) if dummy_dim else (-1, augmult, *image_shape)
        return jnp.reshape(images, target_shape), labels

    if FLAGS.dpsgd and FLAGS.augmult > 0:
        batches = data_stream(aug_train_images, aug_train_labels)
    else:
        batches = data_stream(train_images, train_labels)

    def average_params(params, add_params, t):
        return averaging.average_params(params, add_params, t,
                                        FLAGS.ema_coef, FLAGS.ema_start_step, FLAGS.polyak_start_step)
    opt_init, opt_update, get_params, set_params = sgdavg.sgd(FLAGS.learning_rate)

    rng = random.PRNGKey(42)
    if FLAGS.dataset == "mnist":
        model_kwargs = {'overparameterised': FLAGS.overparameterised, 'groups': FLAGS.groups}
    elif FLAGS.dataset == "cifar10":
        model_kwargs = {'groups': FLAGS.groups}
    else:
        ValueError(f"Unknown dataset: {FLAGS.dataset}")

    model_fn = models.get_model_fn(FLAGS.dataset, model_kwargs)
    model = hk.transform(model_fn, apply_rng=True)

    init_batch = shape_as_image(*next(batches), dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True)[0]
    logger.info(f"Init batch shape: {init_batch.shape}")
    init_args = [init_batch]
    if FLAGS.dataset == "cifar10":
        init_args.append(True)
    init_params = model.init(key, *init_args)
    def predict(params, inputs, is_training=False):
        if FLAGS.dataset == "mnist":
            predictions = model.apply(params, None, inputs)
        elif FLAGS.dataset == "cifar10":
            predictions = model.apply(params, None, inputs, is_training)
        else:
            return ValueError(f"Unknown dataset: {FLAGS.dataset}")
        return predictions

    # jconfig.update('jax_platform_name', 'cpu')

    def ce_loss(params, batch, is_training=False):
      inputs, targets = batch
      logits = predict(params, inputs, is_training=is_training)
      logits = nn.log_softmax(logits, axis=-1)  # log normalize
      return -jnp.mean(jnp.mean(jnp.sum(logits * targets, axis=-1), axis=0))  # cross entropy loss

    def hinge_loss(params, batch, is_training=False):
        inputs, targets = batch
        if len(targets.shape) == 1:
            targets = targets.reshape(1, -1)
        target_class = jnp.argmax(targets, axis=-1)
        scores = predict(params, inputs, is_training=is_training)
        target_class_scores = jnp.choose(target_class, scores.T, mode='wrap')[:, jnp.newaxis]
        return jnp.mean(jnp.mean(jnp.sum(jnp.maximum(0, 1+scores-target_class_scores)-1, axis=-1), axis=0))


    if FLAGS.loss == 'cross-entropy':
        loss = ce_loss
    elif FLAGS.loss == 'hinge':
        loss = hinge_loss
    else:
        raise ValueError("Undefined loss")

    def accuracy(params, batch, splits=1):
      acc = 0
      correct = []
      all_logits = []
      split_size = int(batch[0].shape[0] / splits)
      assert (batch[0].shape[0] % splits) == 0
      for i in range(splits):
        inputs = batch[0][i * split_size:(i + 1) * split_size]
        targets = batch[1][i * split_size:(i + 1) * split_size]
        # logger.info(f"Inputs shape: {inputs.shape}")
        target_class = jnp.argmax(targets, axis=-1)
        logits = predict(params, inputs, is_training=False)
        predicted_class = jnp.argmax(logits, axis=-1)
        # logits_list = logits.tolist()
        # print(logits_list[0])
        acc += jnp.mean(predicted_class == target_class)
        # logger.info("pred class: {}, target class: {}, correct: {}".format(predicted_class.shape, target_class.shape, (predicted_class == target_class).shape))
        correct.append(predicted_class == target_class)
        all_logits.append(logits)
      return acc / splits, jnp.squeeze(jnp.vstack(correct)), jnp.vstack(all_logits)


    def clipped_grad(params, l2_norm_clip, single_example_batch):
      """Evaluate gradient for a single-example batch and clip its grad norm."""
      total_aug_norms = None
      if FLAGS.augmult > 0:
          def single_aug_grad(params, single_aug_batch):
            aug_grads = grad(loss)(params, single_aug_batch, is_training=True)
            nonempty_aug_grads, _ = tree_flatten(aug_grads)
            aug_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_aug_grads])
            return aug_grads, aug_grad_norm
          grads, total_aug_norms = vmap(single_aug_grad, (None, 0))(params, single_example_batch)
          # logger.info(f"Total aug norms: {total_aug_norms}")
          nonempty_grads, tree_def = tree_flatten(grads)
          # aug_norms = jnp.linalg.norm(jnp.hstack([jnp.linalg.norm(g, axis=0) for g in nonempty_grads]), axis=0).tolist()
          nonempty_grads = [g.mean(0) for g in nonempty_grads]
      else:
          grads = grad(loss)(params, single_example_batch, is_training=True)
          nonempty_grads, tree_def = tree_flatten(grads)
      total_grad_norm = jnp.linalg.norm(
          [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
      divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
      normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
      return tree_unflatten(tree_def, normalized_nonempty_grads), total_grad_norm, total_aug_norms


    def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                     batch_size):
      """Return differentially private gradients for params, evaluated on batch."""
      # logger.info("Batch shape: {}".format(batch[0].shape, batch[1].shape))
      clipped_grads, total_grad_norm, total_aug_norms = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
      clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
      aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
      rngs = random.split(rng, len(aggregated_clipped_grads))
      noised_aggregated_clipped_grads = [
          g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
          for r, g in zip(rngs, aggregated_clipped_grads)]
      normalized_noised_aggregated_clipped_grads = [
          g / batch_size for g in noised_aggregated_clipped_grads]
      return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads), total_grad_norm, total_aug_norms


    def non_private_grad(params, batch, batch_size):
        grads, total_grad_norm = vmap(grads_with_norm, (None, None, 0))(params, None, batch)
        grads_flat, grads_treedef = tree_flatten(grads)
        aggregated_grads = [g.sum(0) / batch_size for g in grads_flat]
        return tree_unflatten(grads_treedef, aggregated_grads), total_grad_norm


    def compute_epsilon(steps, num_examples=train_images.shape[0], target_delta=1e-5):
      if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
      q = FLAGS.batch_size / float(num_examples)
      orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
      rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
      eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
      return eps


    def grads_with_norm(params, l2_norm_clip, single_example_batch):
      """Evaluate gradient for a single-example batch and clip its grad norm."""
      grads = grad(loss)(params, single_example_batch, is_training=True)
      nonempty_grads, tree_def = tree_flatten(grads)
      total_grad_norm = jnp.linalg.norm(
          [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
      return tree_unflatten(tree_def, nonempty_grads), total_grad_norm


    def params_norm(params):
        nonempty_params, tree_def = tree_flatten(params)
        total_params_norm = jnp.linalg.norm(
            [jnp.linalg.norm(p.ravel()) for p in nonempty_params]
        )
        # logger.info("Params count:", sum([len(p.ravel()) for p in nonempty_params]))
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
        private_grads, total_grad_norm, total_aug_norms = private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                     FLAGS.noise_multiplier, FLAGS.batch_size)
        opt_state = opt_update(
            i, private_grads, opt_state)
        params = get_params(opt_state)
        avg_params = average_params(params, add_params, i)
        # logger.info(f"Average params: {avg_params}, \n Grads: {private_grads}")
        # logger.info("Optimization state: {}".format(opt_state))
        opt_state = set_params(avg_params, opt_state)
        return opt_state, total_grad_norm, total_aug_norms

    # _, init_params = init_random_params(key, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    grad_norms = []
    aug_norms = []
    param_norms = []
    stats = []
    steps_per_epoch = train_images.shape[0] // FLAGS.batch_size
    add_params = {'ema': get_params(opt_state), 'polyak': get_params(opt_state)}
    logger.info('Starting training...')
    for epoch in range(1, FLAGS.epochs + 1):
        start_time = time.time()
        epoch_grad_norms = []
        epoch_aug_norms = []
        for _ in range(num_batches):
          next_batch = next(batches)
          if FLAGS.dpsgd:
            opt_state, total_grad_norm, total_aug_norms = private_update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params)
          else:
            opt_state, total_grad_norm = update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params)
          acc, correct, logits = accuracy(get_params(opt_state), shape_as_image(*next_batch, augmult=FLAGS.augmult, flatten_augmult=True))
          # print('Grad norm', len(total_grad_norm), 'Correct', len(correct))
          logger.info("Grad norm: {}, Correct: {}, Logits: {}".format(total_grad_norm.shape, correct.shape, logits.shape))
          epoch_grad_norms += zip(total_grad_norm.tolist(), correct.tolist(), logits.tolist())
          if FLAGS.augmult > 0:
            # logger.info(f"Aug norms list: {total_aug_norms.tolist()}")
            epoch_aug_norms += zip(correct.tolist(), total_aug_norms.tolist())
        param_norms.append(float(params_norm(get_params(opt_state))))

        grad_norms.append(epoch_grad_norms)
        aug_norms.append(epoch_aug_norms)
        # log_memory_usage()

        # evaluate test accuracy
        params = get_params(opt_state)
        test_acc, _, _ = accuracy(params, shape_as_image(test_images, test_labels, augmult=0))
        test_loss = loss(params, shape_as_image(test_images, test_labels, augmult=0))
        logger.info('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            test_loss, 100 * test_acc))
        # log_memory_usage()
        train_acc, _, _ = accuracy(params, shape_as_image(train_images, train_labels, augmult=0), splits=5)
        # train_loss = loss(params, shape_as_image(train_images, train_labels, augmult=0))
        train_loss = test_loss
        logger.info('Train set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            train_loss, 100 * train_acc))
        log_memory_usage()

        # determine privacy loss so far
        if FLAGS.dpsgd:
            delta = 1e-5
            num_examples = train_images.shape[0]
            eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
            logger.info(
                'For delta={:.0e}, the current epsilon is: {:.2f}'.format(delta, eps))
        else:
            eps = None
            logger.info('Trained with vanilla non-private SGD optimizer')

        stats.append((train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item(), eps.item()))

        if epoch == FLAGS.epochs:
            if not FLAGS.dpsgd:
                hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_data={FLAGS.dataset},loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging}"
            else:
                hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_data={FLAGS.dataset},loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},nm={FLAGS.noise_multiplier},l2nc={FLAGS.l2_norm_clip},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging},aug={FLAGS.augmult},rf={FLAGS.random_flip},rc={FLAGS.random_crop}"
            with open(ospath.join(FLAGS.norm_dir, f'grad_norms_{hyperparams_string}.pkl'), 'wb') as f:
                # logger.info("grad norms: {}".format(grad_norms[-1][0:100]))
                pickle.dump(grad_norms, f)
            with open(ospath.join(FLAGS.norm_dir, f'param_norms_{hyperparams_string}.pkl'), 'wb') as f:
                # logger.info("param norms: {}".format(param_norms[-1]))
                pickle.dump(param_norms, f)
            with open(ospath.join(FLAGS.norm_dir, f'stats_{hyperparams_string}.pkl'), 'wb') as f:
                pickle.dump(stats, f)
            if FLAGS.augmult > 0:
                with open(ospath.join(FLAGS.norm_dir, f'aug_norms_{hyperparams_string}.pkl'), 'wb') as f:
                    pickle.dump(aug_norms, f)

        epoch_time = time.time() - start_time
        logger.info('Epoch {} in {:0.2f} sec'.format(epoch, epoch_time))
    return hyperparams_string

def main(_):
    if FLAGS.train:
        hyperparams_string = experiment()
    else:
        hyperparams_string = FLAGS.hyperparams_string
    make_plots(hyperparams_string, FLAGS.plot_dir, FLAGS.norm_dir)
    make_single_plot(hyperparams_string, FLAGS.plot_dir)


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e