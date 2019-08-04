import numpy as np
import argparse
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def quantisize(image, levels):
    return np.digitize(image, np.arange(levels) / levels) - 1


def subdict(d, keys):
    return {k:v for k, v in d.items() if k in keys}


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

    save_image(samples, os.path.join(dirname, filename), normalize=True)


def get_loaders(dataset, batch_size, color_levels, train_root, test_root):
    discretize = transforms.Compose([
        transforms.Lambda(lambda image: quantisize(image, color_levels)),
        transforms.ToTensor()
    ])

    to_rgb = transforms.Compose([
        discretize,
        transforms.Lambda(lambda image_tensor: image_tensor.repeat(3, 1, 1))
    ])

    if dataset == "mnist":
        train_dataset = datasets.MNIST(root=train_root, train=True, download=True, transform=to_rgb)
        test_dataset = datasets.MNIST(root=test_root, train=False, download=True, transform=to_rgb)
        HEIGHT, WIDTH = 28, 28
    elif dataset == "fashionmnist":
        train_dataset = datasets.FashionMNIST(root=train_root, train=True, download=True, transform=to_rgb)
        test_dataset = datasets.FashionMNIST(root=test_root, train=False, download=True, transform=to_rgb)
        HEIGHT, WIDTH = 28, 28
    elif dataset == "cifar":
        train_dataset = datasets.CIFAR10(root=train_root, train=True, download=True, transform=discretize)
        test_dataset = datasets.CIFAR10(root=test_root, train=False, download=True, transform=discretize)
        HEIGHT, WIDTH = 32, 32
    else:
        raise AttributeError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return train_loader, test_loader, HEIGHT, WIDTH


def save_checkpoint(state, filename):
    torch.save(state, filename)
