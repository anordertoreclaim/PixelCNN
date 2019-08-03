import numpy as np
import argparse
import os

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets


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


def get_loaders(dataset, transform, batch_size, train_root, test_root):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(root=train_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=test_root, train=False, download=True, transform=transform)
        HEIGHT, WIDTH = 28, 28
    elif dataset == "fashionmnist":
        train_dataset = datasets.FashionMNIST(root=train_root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=test_root, train=False, download=True, transform=transform)
        HEIGHT, WIDTH = 28, 28
    elif dataset == "cifar":
        train_dataset = datasets.CIFAR10(root=train_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=test_root, train=False, download=True, transform=transform)
        HEIGHT, WIDTH = 32, 32
    else:
        raise AttributeError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, test_loader, HEIGHT, WIDTH
