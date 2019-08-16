import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np

import argparse
import os
from utils import str2bool, save_samples, get_loaders

from tqdm import tqdm
import wandb

from pixelcnn import PixelCNN

TRAIN_DATASET_ROOT = '.data/train/'
TEST_DATASET_ROOT = '.data/test/'

MODEL_PARAMS_OUTPUT_DIR = 'model'
MODEL_PARAMS_OUTPUT_FILENAME = 'params.pth'

TRAIN_SAMPLES_DIR = 'train_samples'


def train(cfg, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()

    for images, labels in tqdm(train_loader, desc='Epoch {}/{}'.format(epoch + 1, cfg.epochs)):
        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        normalized_images = images.float() / (cfg.color_levels - 1)

        outputs = model(normalized_images, labels)
        loss = F.cross_entropy(outputs, images)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)

        optimizer.step()

    scheduler.step()


def test_and_sample(cfg, model, device, test_loader, height, width, losses, params, epoch):
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            normalized_images = images.float() / (cfg.color_levels - 1)
            outputs = model(normalized_images, labels)

            test_loss += F.cross_entropy(outputs, images, reduction='none')

    test_loss = test_loss.mean().cpu() / len(test_loader.dataset)

    wandb.log({
        "Test loss": test_loss
    })
    print("Average test loss: {}".format(test_loss))

    losses.append(test_loss)
    params.append(model.state_dict())

    samples = model.sample((3, height, width), cfg.epoch_samples, device=device)
    save_samples(samples, TRAIN_SAMPLES_DIR, 'epoch{}_samples.png'.format(epoch + 1))


def main():
    parser = argparse.ArgumentParser(description='PixelCNN')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to train model for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of images per mini-batch')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to train model on. Either mnist, fashionmnist or cifar.')

    parser.add_argument('--causal-ksize', type=int, default=7,
                        help='Kernel size of causal convolution')
    parser.add_argument('--hidden-ksize', type=int, default=7,
                        help='Kernel size of hidden layers convolutions')

    parser.add_argument('--color-levels', type=int, default=2,
                        help='Number of levels to quantisize value of each channel of each pixel into')

    parser.add_argument('--hidden-fmaps', type=int, default=30,
                        help='Number of feature maps in hidden layer (must be divisible by 3)')
    parser.add_argument('--out-hidden-fmaps', type=int, default=10,
                        help='Number of feature maps in outer hidden layer')
    parser.add_argument('--hidden-layers', type=int, default=6,
                        help='Number of layers of gated convolutions with mask of type "B"')

    parser.add_argument('--learning-rate', '--lr', type=float, default=0.0001,
                        help='Learning rate of optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay rate of optimizer')
    parser.add_argument('--max-norm', type=float, default=1.,
                        help='Max norm of the gradients after clipping')

    parser.add_argument('--epoch-samples', type=int, default=25,
                        help='Number of images to sample each epoch')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='Flag indicating whether CUDA should be used')

    cfg = parser.parse_args()

    wandb.init(project="PixelCNN")
    wandb.config.update(cfg)
    torch.manual_seed(42)

    EPOCHS = cfg.epochs

    model = PixelCNN(cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model.to(device)

    train_loader, test_loader, HEIGHT, WIDTH = get_loaders(cfg.dataset, cfg.batch_size, cfg.color_levels, TRAIN_DATASET_ROOT, TEST_DATASET_ROOT)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, cfg.learning_rate, 10*cfg.learning_rate, cycle_momentum=False)

    wandb.watch(model)

    losses = []
    params = []

    for epoch in range(EPOCHS):
        train(cfg, model, device, train_loader, optimizer, scheduler, epoch)
        test_and_sample(cfg, model, device, test_loader, HEIGHT, WIDTH, losses, params, epoch)

    print('\nBest test loss: {}'.format(np.amin(np.array(losses))))
    print('Best epoch: {}'.format(np.argmin(np.array(losses)) + 1))
    best_params = params[np.argmin(np.array(losses))]

    if not os.path.exists(MODEL_PARAMS_OUTPUT_DIR):
        os.mkdir(MODEL_PARAMS_OUTPUT_DIR)
    MODEL_PARAMS_OUTPUT_FILENAME = '{}_cks{}hks{}cl{}hfm{}ohfm{}hl{}_params.pth'\
        .format(cfg.dataset, cfg.causal_ksize, cfg.hidden_ksize, cfg.color_levels, cfg.hidden_fmaps, cfg.out_hidden_fmaps, cfg.hidden_layers)
    torch.save(best_params, os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME))


if __name__ == '__main__':
    main()
