import numpy as np
import argparse
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def save_samples(samples, dirname, filename):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    count = samples.size()[0]

    count_sqrt = int(count ** 0.5)
    if count_sqrt ** 2 == count:
        nrow = count_sqrt
    else:
        nrow = count

    save_image(samples, os.path.join(dirname, filename), nrow=nrow)


def get_loaders(dataset_name, batch_size, color_levels, train_root, test_root):
    normalize = transforms.Lambda(lambda image: np.array(image) / 255)

    discretize = transforms.Compose([
        transforms.Lambda(lambda image: quantisize(image, color_levels)),
        transforms.ToTensor()
    ])

    to_rgb = transforms.Compose([
        discretize,
        transforms.Lambda(lambda image_tensor: image_tensor.repeat(3, 1, 1))
    ])

    dataset_mappings = {'mnist': 'MNIST', 'fashionmnist': 'FashionMNIST', 'cifar': 'CIFAR10'}
    transform_mappings = {'mnist': to_rgb, 'fashionmnist': to_rgb, 'cifar': transforms.Compose([normalize, discretize])}
    hw_mappings = {'mnist': (28, 28), 'fashionmnist': (28, 28), 'cifar': (32, 32)}

    try:
        dataset = dataset_mappings[dataset_name]
        transform = transform_mappings[dataset_name]

        train_dataset = getattr(datasets, dataset)(root=train_root, train=True, download=True, transform=transform)
        test_dataset = getattr(datasets, dataset)(root=test_root, train=False, download=True, transform=transform)

        h, w = hw_mappings[dataset_name]
    except KeyError:
        raise AttributeError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    return train_loader, test_loader, h, w
