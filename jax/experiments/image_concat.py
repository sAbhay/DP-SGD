from sys import path as syspath
syspath.append('../')

from common import log
logger = log.get_logger('plot_concat')

import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
from analysis import NORM_DIR, get_hyperparameter_strings, PLOTS_DIR

HYPERPARAMETER_STRING_FORMS = {
    "dataset": "Dataset",
    "model": "Model",
    "depth": "Depth",
    "loss": "Loss",
    "lr": "Learning rate",
    "batch_size": "Batch size",
    "op": "Overparameterised",
    "nm": "Noise multiplier",
    "l2nc": "L2 norm clip",
    "grp": "Groups",
    "ws": "Weight standardisation",
    "pa": "Parameter averaging",
    "ess": "EMA start step",
    "pss": "Polyak start step",
    "mu": "EMA coefficient",
    "aug": "Augmentations",
    "rf": "Random flip",
    "rc": "Random crop",
}


def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS)
              for image in images]

    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            if idx >= len(images):
                continue
            image.paste(images[idx], offset)

    return image


def make_single_plot(hyperparameter_string, plot_dir):
    folder = os.path.join(plot_dir, hyperparameter_string)
    image_paths = [os.path.join(folder, f)
                   for f in os.listdir(folder) if f.endswith('.png')]

    # Create and save image grid
    image = concat_images(image_paths, (640, 480), (2, 3))
    title_plot(image, hyperparameter_string)
    image.save(os.path.join(plot_dir, 'all', f'{hyperparameter_string}.png'), 'PNG')

def title_plot(image, hyperparam_string):
    image = ImageOps.expand(image, border=45, fill=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    splits = hyperparam_string.split('_')
    title = splits[0] + "with "
    splits = splits[1].split(',')
    for split in splits:
        k, v = split.split('=')
        if k in {"dataset", "model", "depth", "batch_size", "grp", "ws", "pa", "aug"}:
            title += HYPERPARAMETER_STRING_FORMS[k] + "=" + v + " "
    logger.log("title: {}".format(title))

    draw.text((0, 0), title, (0, 0, 0), font=ImageFont.load_default())


if __name__ == '__main__':
    hyperparameter_strings = get_hyperparameter_strings(NORM_DIR)

    while len(hyperparameter_strings) > 0:
        hyperparameter_string = hyperparameter_strings.pop()
        make_single_plot(hyperparameter_string, PLOTS_DIR)