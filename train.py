import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

import argparse
import os
import sys
from utils import str2bool, quantisize, save_samples, get_loaders
from tqdm import tqdm
import wandb

from pixelcnn import PixelCNN

TRAIN_DATASET_ROOT = ".data/train/"
TEST_DATASET_ROOT = ".data/test/"

MODEL_OUTPUT_DIR = "model"

TRAIN_SAMPLES_DIRNAME = "train_samples"
TRAIN_SAMPLES_COUNT = 9


def train(cfg, model, device, train_loader, optimizer, epoch):
    model.train()

    for images, _ in tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, cfg.epochs)):
        optimizer.zero_grad()

        images = images.to(device)
        normalized_images = images.float() / (cfg.color_levels - 1)

        outputs = model(normalized_images)
        loss = F.cross_entropy(outputs, images)
        loss.backward()
        optimizer.step()


def test_and_sample(cfg, model, device, test_loader, height, width, epoch):
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            normalized_images = images.float() / (cfg.color_levels - 1)
            outputs = model(normalized_images)

            test_loss += F.cross_entropy(outputs, images)

    test_loss /= len(test_loader.dataset)

    wandb.log({
        "Test loss": test_loss
    })
    print("\nAverage test loss: {}".format(test_loss))

    samples = model.sample((cfg.data_channels, height, width), TRAIN_SAMPLES_COUNT, device=device)
    save_samples(samples, TRAIN_SAMPLES_DIRNAME, 'epoch{}_samples.png'.format(epoch))


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
    parser.add_argument('--model-output-fname', '-m', default='params.pth',
                        help="Output path for model's parameters")
    parser.add_argument('--samples-folder', '-o', type=str, default='train-samples/',
                        help='Path where sampled images will be saved')

    cfg = parser.parse_args()

    wandb.init(project="PixelCNN")
    wandb.config.update(cfg)

    LEVELS = cfg.color_levels
    EPOCHS = cfg.epochs

    model = PixelCNN(cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: quantisize(image, LEVELS)),
        transforms.ToTensor(),
    ])

    train_loader, test_loader, HEIGHT, WIDTH = get_loaders(cfg.dataset, transform, cfg.batch_size, TRAIN_DATASET_ROOT, TEST_DATASET_ROOT)

    optimizer = optim.Adam(model.parameters())

    wandb.watch(model)

    for epoch in range(EPOCHS):
        train(cfg, model, device, train_loader, optimizer, epoch)
        test_and_sample(cfg, model, device, test_loader, HEIGHT, WIDTH, epoch)

    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.mkdir(MODEL_OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, cfg.model_output_fname))


if __name__ == '__main__':
    sys.exit(main())
