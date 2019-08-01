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
    samples = samples.view(count**0.5, count**0.5, channels, height, width)
    samples = samples.permute(1, 3, 0, 4, 2)
    samples = samples.view(height * count, width * count, channels) * 255
    Image.fromarray(samples).save(filename)
