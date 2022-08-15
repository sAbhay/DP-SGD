from sys import path as syspath
syspath.append('../')

from common import log
logger = log.get_logger('util')

from os import path as ospath
import pickle
import psutil
import nvidia_smi

from analysis import make_plots
from image_concat import make_single_plot


def get_hyperparameter_string(FLAGS):
    if not FLAGS.dpsgd:
        hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_data={FLAGS.dataset},model={FLAGS.model},depth={FLAGS.depth},width={FLAGS.width},loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging}"
    else:
        hyperparams_string = f"{'dpsgd' if FLAGS.dpsgd else 'sgd'}_data={FLAGS.dataset},model={FLAGS.model},depth={FLAGS.depth},width={FLAGS.width},loss={FLAGS.loss},lr={FLAGS.learning_rate},op={FLAGS.overparameterised},nm={FLAGS.noise_multiplier},l2nc={FLAGS.l2_norm_clip},grp={FLAGS.groups},bs={FLAGS.batch_size},ws={FLAGS.weight_standardisation},mu={FLAGS.ema_coef},ess={FLAGS.ema_start_step},pss={FLAGS.polyak_start_step},pa={FLAGS.param_averaging},augtype={FLAGS.aug_type},aug={FLAGS.augmult},rf={FLAGS.random_flip},rc={FLAGS.random_crop},mr={FLAGS.mult_radius},mass={FLAGS.mass},pb={FLAGS.privacy_budget},adnc={FLAGS.adaptive_norm_clip}"
    return hyperparams_string


def plot_results(hyperparams_string, plot_dir, norm_dir):
    try:
        make_plots(hyperparams_string, plot_dir, norm_dir)
    except (Exception, FileNotFoundError) as e:
        logger.error(e, exc_info=True)
        raise e
    make_single_plot(hyperparams_string, plot_dir)


def checkpoint(FLAGS, grad_norms, param_norms, stats, aug_norms=None, plot=False):
    hyperparams_string = get_hyperparameter_string(FLAGS)
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

    if plot:
        plot_results(hyperparams_string, FLAGS.plot_dir, FLAGS.norm_dir)

    return hyperparams_string


def log_memory_usage(logger, handle):
    logger.info(f"RAM usage: {psutil.virtual_memory()}")
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    logger.info(f'GPU usage: {100 * (mem_res.used / mem_res.total):.3f}%')

