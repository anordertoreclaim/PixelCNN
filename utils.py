import numpy as np
import argparse

from PIL import Image


def quantisize(image, levels):
    return np.digitize(image, np.arange(levels) / levels) - 1


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def nearest_square(num):
    return round(np.sqrt(num))**2


def save_samples(samples, filename):
    count, channels, height, width = samples.size()
    images_on_side = int(count ** 0.5)
    samples = samples.view(images_on_side, images_on_side, channels, height, width)
    samples = samples.permute(1, 3, 0, 4, 2).contiguous()
    samples = samples.view(height * images_on_side, width * images_on_side, channels) * 255
    samples = samples.squeeze()
    Image.fromarray(samples).save(filename)
