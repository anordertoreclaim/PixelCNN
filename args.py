import argparse
from utils import *

def getCFG():
	parser = argparse.ArgumentParser(description='PixelCNN')

	parser.add_argument('--epochs', type=int, default=25,
						help='Number of epochs to train model for')
	parser.add_argument('--batch-size', type=int, default=32,
						help='Number of images per mini-batch')
	parser.add_argument('--dataset', type=str, default='mnist',
						help='Dataset to train model on. Either mnist, fashionmnist, cifar or celeba.')

	parser.add_argument('--causal-ksize', type=int, default=7,
						help='Kernel size of causal convolution')
	parser.add_argument('--hidden-ksize', type=int, default=7,
						help='Kernel size of hidden layers convolutions')

	parser.add_argument('--color-levels', type=int, default=2,
						help='Number of levels to quantisize value of each channel of each pixel into')

	parser.add_argument('--classes', type=int, default=10,
						help="number of classes in the dataset")
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

	parser.add_argument('--epoch-samples', type=int, default=9,
						help='Number of images to sample each epoch')

	parser.add_argument('--dataset-size', type=int,
						help='Number of max images to use for dataset sizes (Note: only implemented for CelebA)')

	parser.add_argument('--use-artifact', type=str,
						help="artifact name from wandb to use instead of default model") # TODO: test

	parser.add_argument('--cuda', type=str2bool, default=True,
						help='Flag indicating whether CUDA should be used')
	parser.add_argument('--upload', action="store_true",
						help='Flag indicating whether or not to upload each train to wandb')
	parser.add_argument('--overfit', action="store_true",
						help='If this flag is set, train will be set to whatever test is')
	parser.add_argument('--save-sample', action="store_true",
						help='If this flag is set, it will save an image from train/test to disk.')

	cfg = parser.parse_args()
	if cfg.dataset == "celeba":
		cfg.classes = 1
	return cfg