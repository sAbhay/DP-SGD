# Largely from https://github.com/deepmind/jax_privacy/blob/main/jax_privacy/src/training/image_classification/data/augmult.py

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

"""Data augmentation with augmult (Hoffer et al., 2019; Fort et al., 2021).

References:
  E. Hoffer, T. Ben-Nun, I. Hubara, N. Giladi, T. Hoefler, and D. Soudry.
  Augment your batch: bettertraining with larger batches.arXiv, 2019.
  S. Fort, A. Brock, R. Pascanu, S. De, and S. L. Smith.
  Drawing multiple augmentation samples perimage during training efficiently
  decreases test error.arXiv, 2021.
"""
from sys import path
path.append('.')
from common import log
logger = log.get_logger('augmult')

from typing import Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  # try:
  #   tf.config.experimental.set_virtual_device_configuration(gpu, [
  #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  # except RuntimeError as e:
  #   logger.error(e)

def apply_augmult_tf(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: Sequence[int],
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    crop_size: Optional[Sequence[int]] = None,
    pad: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Augmult data augmentation (Hoffer et al., 2019; Fort et al., 2021).

  Args:
    image: (single) image to augment.
    label: label corresponding to the image (not modified by this function).
    image_size: new size for the image.
    augmult: number of augmentation multiplicities to use. This number should
      be non-negative (this function will fail if it is not).
    random_flip: whether to use random horizontal flips for data augmentation.
    random_crop: whether to use random crops for data augmentation.
    crop_size: size of the crop for random crops.
    pad: optional padding before the image is cropped.
  Returns:
    images: augmented images with a new prepended dimension of size `augmult`.
    labels: repeated labels with a new prepended dimension of size `augmult`.
  """
  image = tf.reshape(image, image_size)

  # No augmentations; return original images and labels with a new dimension.
  if augmult == 0:
    images = tf.expand_dims(image, axis=0)
    labels = tf.expand_dims(label, axis=0)
  # Perform one or more augmentations.
  elif augmult > 0:
    raw_image = tf.identity(image)
    augmented_images = []

    for _ in range(augmult):
      image_now = raw_image

      if random_crop:
        if pad:
          image_now = padding_input(image_now, pad=pad)
        image_now = tf.image.random_crop(
            image_now,
            size=(crop_size if crop_size is not None else image_size),
        )
      if random_flip:
        image_now = tf.image.random_flip_left_right(image_now)

      augmented_images.append(image_now)

    images = tf.stack(augmented_images, axis=0)
    labels = tf.stack([label]*augmult, axis=0)
  else:
    raise ValueError('Augmult should be non-negative.')

  return images, labels


def padding_input(x: tf.Tensor, pad: int):
  """Pad input image through 'mirroring' on the four edges.

  Args:
    x: image to pad.
    pad: number of padding pixels.
  Returns:
    Padded image.
  """
  x = tf.concat([x[:pad, :, :][::-1], x, x[-pad:, :, :][::-1]], axis=0)
  x = tf.concat([x[:, :pad, :][:, ::-1], x, x[:, -pad:, :][:, ::-1]], axis=1)
  return x

def apply_augmult_single(
          image: np.ndarray,
          label: np.ndarray,
          *,
          image_size: Sequence[int],
          augmult: int,
          random_flip: bool,
          random_crop: bool,
          crop_size: Optional[Sequence[int]] = None,
          pad: Optional[int] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:

    aug_images, aug_labels = apply_augmult_tf(image, label,
                                              image_size=image_size, augmult=augmult, random_flip=random_flip,
                                              random_crop=random_crop, crop_size=crop_size, pad=pad)
    return aug_images.numpy(), aug_labels.numpy()


def apply_augmult(
        images: np.ndarray,
        labels: np.ndarray,
        *,
        image_size: Sequence[int],
        augmult: int,
        random_flip: bool,
        random_crop: bool,
        crop_size: Optional[Sequence[int]] = None,
        pad: Optional[int] = None,
        batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
  ret_images = np.zeros((images.shape[0], augmult, images.shape[1]))
  ret_labels = np.zeros((labels.shape[0], augmult, labels.shape[1]))
  num_batches = int(np.ceil(images.shape[0] / batch_size))
  def apply_augmult_partial(args):
    images, labels = args
    return apply_augmult_tf(images, labels, image_size=image_size, augmult=augmult,
                                                  random_flip=random_flip, random_crop=random_crop, crop_size=crop_size,
                                                  pad=pad)
  for i in range(num_batches):
    batch_idx = range(i * batch_size, (i + 1) * batch_size)
    if i == num_batches - 1:
      batch_idx = range(i * batch_size, images.shape[0])
    batch_images_tf = tf.convert_to_tensor(images[batch_idx])
    batch_labels_tf = tf.convert_to_tensor(labels[batch_idx])
    batch_images_tf, batch_labels_tf = tf.vectorized_map(apply_augmult_partial, (batch_images_tf, batch_labels_tf))
    # logger.info(f"augmult images: {images.shape}, labels: {labels.shape}, image: {images[0][:10]}, label: {labels[0]}")
    ret_images[batch_idx] = batch_images_tf.numpy().reshape((batch_images_tf.shape[0], batch_images_tf.shape[1], -1))
    ret_labels[batch_idx] = batch_labels_tf.numpy()
  return ret_images, ret_labels
