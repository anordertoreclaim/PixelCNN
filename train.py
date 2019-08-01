import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse
import os
from utils import str2bool, quantisize, save_samples
from tqdm import tqdm

from pixelcnn import PixelCNN

DATASET_ROOT = "data/"
TRAIN_SAMPLES_PATH = "train_samples"
TRAIN_SAMPLES_COUNT = 16 #must be square


def main():
    parser = argparse.ArgumentParser(description='PixelCNN')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train model for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of images per mini-batch')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to train model on. Either mnist, fashionmnist or cifar.')

    parser.add_argument('--causal-ksize', type=int, default=7,
                        help='Kernel size of causal convolution')
    parser.add_argument('--hidden-ksize', type=int, default=3,
                        help='Kernel size of hidden layers convolutions')

    parser.add_argument('--data-channels', type=int, default=1,
                        help='Number of data channels')
    parser.add_argument('--color-levels', type=int, default=7,
                        help='Number of levels to quantisize value of each channel of each pixel into')

    parser.add_argument('--hidden-fmaps', type=int, default=128,
                        help='Number of feature maps in hidden layer')
    parser.add_argument('--out-hidden-fmaps', type=int, default=32,
                        help='Number of feature maps in outer hidden layer')
    parser.add_argument('--hidden-layers', type=int, default=10,
                        help='Number of layers of gated convolutions with mask of type "B"')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='Flag indicating whether CUDA should be used')
    parser.add_argument('--model-output-path', '-m', default='',
                        help="Output path for model's parameters")
    parser.add_argument('--samples-folder', '-o', type=str, default='train-samples/',
                        help='Path where sampled images will be saved')

    cfg = parser.parse_args()
    LEVELS = cfg.color_levels
    MODEL_PATH = cfg.model_output_path

    model = PixelCNN(cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: quantisize(image, LEVELS)),
        transforms.ToTensor()
    ])
    if cfg.dataset == "mnist":
        dataset = datasets.MNIST(root=DATASET_ROOT, train=True, download=True, transform=transform)
        HEIGHT, WIDTH = 28, 28
    elif cfg.dataset == "fashionmnist":
        dataset = datasets.FashionMNIST(root=DATASET_ROOT, train=True, download=True, transform=transform)
        HEIGHT, WIDTH = 28, 28
    elif cfg.dataset == "cifar":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True, transform=transform)
        HEIGHT, WIDTH = 28, 28

    data_loader = DataLoader(dataset, batch_size=cfg.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in tqdm(range(cfg.epochs)):
        for i, images in enumerate(data_loader):
            optimizer.zero_grad()

            if cfg.dataset in ['mnist', 'fashionmnist', 'cifar']:
                # remove labels
                images = images[0]

            normalized_images = images.float() / (LEVELS - 1)

            outputs = model(normalized_images)
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()

        model.eval()
        samples = model.sample((cfg.data_channels, HEIGHT, WIDTH), TRAIN_SAMPLES_COUNT)
        save_samples(samples, os.path.join(TRAIN_SAMPLES_PATH, 'epoch{}_samples.jpg'.format(epoch)))
        model.train()

    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    main()