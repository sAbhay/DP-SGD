from sys import path
path.append('../')

from common import log
logger = log.get_logger('main')

from absl import app
from absl import flags

from experiment import experiment

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
    logger.info(FLAGS)
    experiment(FLAGS)


if __name__ == '__main__':
    try:
        logger.info(FLAGS)
        app.run(main)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e