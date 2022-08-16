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
from functools import partial
from sys import path as syspath

syspath.append('../')

from common import log
logger = log.get_logger('experiment')

import itertools
import time

from jax import jit
from jax import random
from jax import nn
import jax.numpy as jnp
import haiku as hk
import jax
from jax import pmap

import numpy.random as npr

import nvidia_smi

from data import datasets
import models.models as models
from optim.optimizer.sgd import sgdavg
from optim.optimizer.momentum import sgd_momentum_avg
from optim.grad import non_private as np_grad
from optim.grad.private import private, aug_data, aug_momentum
from optim import update as up
from common import averaging
from common import util as cutil

from absl import app
from absl import flags

from util import plot_results, checkpoint, get_hyperparameter_string, log_memory_usage

FLAGS = flags.FLAGS

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

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
flags.DEFINE_string('model', "cnn", "Model: cnn, wideresnet, or wideresnet2")
flags.DEFINE_integer('depth', 6, "Network depth")
flags.DEFINE_integer('checkpoint', 20, "Checkpoint interval in epochs")
flags.DEFINE_integer('width', 1, "Network width")
flags.DEFINE_string('aug_type', 'data', 'Augmentation type: data or momentum')
flags.DEFINE_float('mult_radius', 1, 'Radius for momentum sampling')
flags.DEFINE_float('mass', 0.9, 'Momentum coefficient')
flags.DEFINE_float('privacy_budget', None, 'Privacy budget')
flags.DEFINE_boolean('adaptive_norm_clip', False, 'Adaptive l2 norm clip')

def experiment():
    logger.info("Running Experiment")
    log_memory_usage(logger, handle)
    num_devices = jax.local_device_count()
    logger.info("Found {} devices".format(num_devices))

    if FLAGS.microbatches:
        raise NotImplementedError(
            'Microbatches < batch size not currently supported'
        )

    train_images, train_labels, test_images, test_labels = datasets.data(name=FLAGS.dataset)
    logger.info(f"Train set shape: {train_images.shape}, {train_labels.shape}")
    log_memory_usage(logger, handle)
    if FLAGS.dpsgd and FLAGS.aug_type == 'data' and FLAGS.augmult > 0:
        start_time = time.time()
        image_size = datasets.IMAGE_SHAPE[FLAGS.dataset]
        aug_train_images, aug_train_labels = datasets.apply_augmult(train_images, train_labels,
                                                            image_size=image_size, augmult=FLAGS.augmult,
                                                            random_flip=FLAGS.random_flip, random_crop=FLAGS.random_crop,
                                                            batch_size=FLAGS.augmult_batch_size)
        logger.info(f"Augmented train set shape: {aug_train_images.shape}, {aug_train_labels.shape}")
        logger.info("Augmented train images in {:.2f} sec".format(time.time() - start_time))
        log_memory_usage(logger, handle)
    else:
        logger.info("No data augmentation applied")
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

    def make_superbatch(input_dataset):
        """Constructs a superbatch, i.e. one batch of data per device."""
        # Get N batches, then split into list-of-images and list-of-labels.
        superbatch = [next(input_dataset) for _ in range(num_devices)]
        superbatch_images, superbatch_labels = zip(*superbatch)
        # Stack the superbatches to be one array with a leading dimension, rather than
        # a python list. This is what `jax.pmap` expects as input.
        superbatch_images = jnp.stack(superbatch_images)
        superbatch_labels = jnp.stack(superbatch_labels)
        return superbatch_images, superbatch_labels

    def shape_as_image(images, labels, dataset=FLAGS.dataset, dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True, aug_type=FLAGS.aug_type):
        # logger.info(f"Preshaped images shape: {images.shape}")
        image_shape = datasets.IMAGE_SHAPE[dataset]
        target_shape = (num_devices, -1, 1, *image_shape) if dummy_dim else (num_devices, -1, *image_shape)
        if flatten_augmult:
            if augmult > 0 and aug_type == 'data':
                # logger.info(f"Preshaped labels shape: {labels.shape}")
                labels = jnp.reshape(labels, (num_devices, -1, *labels.shape[2:]))
        elif augmult > 0 and aug_type == 'data':
            target_shape = (num_devices, -1, augmult, 1, *image_shape) if dummy_dim else (num_devices, -1, augmult, *image_shape)
        return jnp.reshape(images, target_shape), jnp.reshape(labels, (num_devices, *labels.shape[1:]))

    if FLAGS.dpsgd and FLAGS.aug_type == 'data' and FLAGS.augmult > 0:
        batches = data_stream(aug_train_images, aug_train_labels)
    else:
        batches = data_stream(train_images, train_labels)

    def average_params(params, add_params, t):
        return averaging.average_params(params, add_params, t,
                                        FLAGS.ema_coef, FLAGS.ema_start_step, FLAGS.polyak_start_step)
    opt_init, opt_update, get_params, get_velocity, set_params = sgd_momentum_avg.sgd_momentum(FLAGS.learning_rate, FLAGS.mass)

    rng = random.PRNGKey(42)
    if FLAGS.model == "cnn":
        model_kwargs = {'overparameterised': FLAGS.overparameterised, 'groups': FLAGS.groups,
                        'weight_standardisation': FLAGS.weight_standardisation, 'depth': FLAGS.depth}
    elif FLAGS.model == "wideresnet" or FLAGS.model == "wideresnet2":
        model_kwargs = {'groups': FLAGS.groups, 'depth': FLAGS.depth, 'width': FLAGS.width}
    else:
        ValueError(f"Unknown model: {FLAGS.model}")

    model_fn = models.get_model_fn(FLAGS.model, model_kwargs)
    model = hk.transform(model_fn, apply_rng=True)

    init_batch = (jnp.zeros((FLAGS.batch_size, *datasets.IMAGE_SHAPE[FLAGS.dataset])), jnp.zeros(train_labels[0:FLAGS.batch_size].shape))[0]
    # init_batch = shape_as_image(*dummy_batch, dummy_dim=False, augmult=FLAGS.augmult, flatten_augmult=True)[0]
    logger.info(f"Init batch shape: {init_batch.shape}")
    init_args = [init_batch]
    if FLAGS.model == "wideresnet":
        init_args.append(True)
    init_params = model.init(key, *init_args)

    def predict(params, inputs):
        if FLAGS.model == "cnn":
            predictions = pmap(model.apply, axis_name='i')(params, inputs)
        elif FLAGS.model == "wideresnet" or FLAGS.model == "wideresnet2":
            predictions = pmap(model.apply, axis_name='i')(params, inputs)
        else:
            return ValueError(f"Unknown model: {FLAGS.model}")
        return predictions

    # jconfig.update('jax_platform_name', 'cpu')

    def ce_loss(params, batch):
      inputs, targets = batch
      logits = predict(params, inputs)
      logits = nn.log_softmax(logits, axis=-1)  # log normalize
      return -jnp.mean(jnp.mean(jnp.sum(logits * targets, axis=-1), axis=0))  # cross entropy loss

    def hinge_loss(params, batch):
        inputs, targets = batch
        if len(targets.shape) == 1:
            targets = targets.reshape(1, -1)
        target_class = jnp.argmax(targets, axis=-1)
        scores = predict(params, inputs)
        target_class_scores = jnp.choose(target_class, scores.T, mode='wrap')[:, jnp.newaxis]
        return jnp.mean(jnp.mean(jnp.sum(jnp.maximum(0, 1+scores-target_class_scores)-1, axis=-1), axis=0))


    if FLAGS.loss == 'cross-entropy':
        loss = ce_loss
    elif FLAGS.loss == 'hinge':
        loss = hinge_loss
    else:
        raise ValueError("Undefined loss")

    # @functools.partial(pmap, axis_name='i', donate_argnums=(0,))
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
        logits = predict(params, inputs)
        predicted_class = jnp.argmax(logits, axis=-1)
        # logits_list = logits.tolist()
        # print(logits_list[0])
        acc += jnp.mean(predicted_class == target_class)
        # logger.info("pred class: {}, target class: {}, correct: {}".format(predicted_class.shape, target_class.shape, (predicted_class == target_class).shape))
        correct.append(predicted_class == target_class)
        all_logits.append(logits)
      return acc / splits, jnp.squeeze(jnp.vstack(correct)), jnp.vstack(all_logits)


    def update(_, i, opt_state, batch, add_params):
        params = get_params(opt_state)
        grads, total_grad_norm = np_grad.non_private_grad(params, batch, FLAGS.batch_size, loss)
        opt_state = opt_update(i, grads, opt_state)
        if FLAGS.param_averaging:
            params = get_params(opt_state)
            avg_params = average_params(params, add_params, i)
            opt_state = set_params(avg_params, opt_state)
        return opt_state, total_grad_norm


    def private_update(rng, i, opt_state, batch, add_params, l2_norm_clip=FLAGS.l2_norm_clip):
        params = get_params(opt_state)
        rng = random.fold_in(rng, i)  # get new key for new random numbers
        rng = jnp.broadcast_to(rng, (num_devices,) + rng.shape)
        if FLAGS.augmult <= 0:
            private_grads, total_grad_norm = pmap(private.private_grad, axis_name='i')(params, batch, rng, l2_norm_clip,
                                                                           FLAGS.noise_multiplier, FLAGS.batch_size,
                                                                           loss)
            total_aug_norms = None
        elif FLAGS.aug_type == "data":
            private_grads, total_grad_norm, total_aug_norms = pmap(aug_data.private_grad, axis_name='i')(params, batch, rng,
                                                                                   l2_norm_clip,
                                                                                   FLAGS.noise_multiplier,
                                                                                   FLAGS.batch_size,
                                                                                   loss)
        elif FLAGS.aug_type == "momentum":
            velocity = get_velocity(opt_state)
            private_grads, total_grad_norm, total_aug_norms = \
                pmap(partial(aug_momentum.private_grad, l2_norm_clip=l2_norm_clip, noise_multiplier=FLAGS.noise_multiplier,
                             batch_size=FLAGS.batch_size, loss=loss, augmult=FLAGS.augmult, mult_radius=FLAGS.mult_radius), axis_name='i')\
                    (params, batch, rng, velocity)
        else:
            raise ValueError("Undefined augmentation type")
        private_grads = jax.lax.pmean(private_grads, axis_name='i')
        opt_state = opt_update(
            i, private_grads, opt_state)
        if FLAGS.param_averaging:
            params = get_params(opt_state)
            avg_params = average_params(params, add_params, i)
            opt_state = set_params(avg_params, opt_state)
        return opt_state, total_grad_norm, total_aug_norms

    # _, init_params = init_random_params(key, (-1, 28, 28, 1))
    # logger.info("Model init params: {}".format(init_params))
    up.params_norm(init_params)
    init_params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), init_params)
    logger.info("Model init params shape: {}".format(cutil.params_shape(init_params)))
    opt_state = opt_init(init_params)
    logger.info("Model params shape: {}".format(cutil.params_shape(get_params(opt_state))))
    itercount = itertools.count()

    epoch_average_grad_norm = 0
    grad_norms = []
    aug_norms = []
    param_norms = []
    stats = []
    steps_per_epoch = train_images.shape[0] // FLAGS.batch_size
    add_params = {'ema': get_params(opt_state), 'polyak': get_params(opt_state)}
    logger.info('Starting training...')
    l2_norm_clip = FLAGS.l2_norm_clip
    for epoch in range(1, FLAGS.epochs + 1):
        start_time = time.time()
        epoch_grad_norms = []
        epoch_aug_norms = []
        for _ in range(num_batches // num_devices):
          # next_batch = next(batches)
          next_batch = make_superbatch(batches)
          if FLAGS.dpsgd:
            opt_state, total_grad_norm, total_aug_norms = private_update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params, l2_norm_clip)
          else:
            opt_state, total_grad_norm = update(
                key, next(itercount), opt_state, shape_as_image(*next_batch, dummy_dim=True, augmult=FLAGS.augmult, flatten_augmult=False), add_params)
          acc, correct, logits = accuracy(get_params(opt_state), shape_as_image(*next_batch, augmult=FLAGS.augmult, flatten_augmult=True))
          epoch_grad_norms += zip(total_grad_norm.tolist(), correct.tolist(), logits.tolist())
          epoch_average_grad_norm += sum(total_grad_norm.tolist())

          if FLAGS.augmult > 0:
            # logger.info(f"Aug norms list: {total_aug_norms.tolist()}")
            epoch_aug_norms += zip(correct.tolist(), total_aug_norms.tolist())
        param_norms.append(float(up.params_norm(get_params(opt_state))))

        grad_norms.append(epoch_grad_norms)
        aug_norms.append(epoch_aug_norms)
        # log_memory_usage(logger, handle)
        epoch_average_grad_norm /= train_images.shape[0]
        logger.info(f"Epoch average grad norm: {epoch_average_grad_norm}")
        if FLAGS.adaptive_norm_clip:
            l2_norm_clip = epoch_average_grad_norm

        # evaluate test accuracy
        params = get_params(opt_state)
        test_acc, _, _ = accuracy(params, shape_as_image(test_images, test_labels, augmult=0), splits=5)
        test_loss = loss(params, shape_as_image(test_images, test_labels, augmult=0))
        logger.info('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            test_loss, 100 * test_acc))
        # log_memory_usage(logger, handle)
        train_acc, _, _ = accuracy(params, shape_as_image(train_images, train_labels, augmult=0), splits=5)
        # train_loss = loss(params, shape_as_image(train_images, train_labels, augmult=0))
        train_loss = test_loss
        logger.info('Train set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
            train_loss, 100 * train_acc))
        log_memory_usage(logger, handle)

        # determine privacy loss so far
        if FLAGS.dpsgd:
            delta = 1e-5
            num_examples = train_images.shape[0]
            eps = up.compute_epsilon(epoch * steps_per_epoch, num_examples, FLAGS.batch_size, FLAGS.noise_multiplier, delta)
            logger.info(
                'For delta={:.0e}, the current epsilon is: {:.2f}'.format(delta, eps))
        else:
            eps = None
            logger.info('Trained with vanilla non-private SGD optimizer')

        stats.append((train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item(), eps.item() if FLAGS.dpsgd else None))

        if (epoch % FLAGS.checkpoint) == 0 or epoch == FLAGS.epochs - 1:
            hyperparams_string = checkpoint(FLAGS, grad_norms, param_norms, stats, aug_norms=aug_norms, plot=True)

        epoch_time = time.time() - start_time
        logger.info('Epoch {} in {:0.2f} sec'.format(epoch, epoch_time))
        if FLAGS.privacy_budget is not None and eps >= FLAGS.privacy_budget:
            logger.info('Privacy budget exceeded!')
            break
    return hyperparams_string

def main(_):
    hyperparams_string = None
    if FLAGS.train:
        try:
            hyperparams_string = experiment()
        except KeyboardInterrupt:
            logger.warn("Experiment interrupted by user, saving plots")
    if hyperparams_string is None:
        if FLAGS.hyperparams_string is not None:
            hyperparams_string = FLAGS.hyperparams_string
        else:
            hyperparams_string = get_hyperparameter_string(FLAGS)

    plot_results(hyperparams_string, FLAGS.plot_dir, FLAGS.norm_dir)


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e