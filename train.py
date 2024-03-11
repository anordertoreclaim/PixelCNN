import torch
import pickle
import deeplake
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np

from args import getCFG
import os
from utils import *

from tqdm import tqdm
import wandb

# This is to hack some issue with PIL images - see
# https://wandb.ai/pixelcnn/PixelCNN/runs/lfa9h4h9
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pixelcnn import PixelCNN

TRAIN_DATASET_ROOT = '.data/train/'
TEST_DATASET_ROOT = '.data/test/'

MODEL_PARAMS_OUTPUT_DIR = 'model'
MODEL_PARAMS_OUTPUT_FILENAME = 'params.pth'

TRAIN_SAMPLES_DIR = 'train_samples'

cfg = getCFG()

def train(cfg, model, device, train_loader, optimizer, scheduler, epoch):
	model.train()
	HAS_LABELS = None

	saveflag=cfg.save_sample # turn on to save a batch
	for data in tqdm(train_loader, desc='Epoch {}/{}'.format(epoch + 1, cfg.epochs).ljust(20)):
		if HAS_LABELS is None:
			HAS_LABELS=True
			try:
				images, labels = data
			except ValueError:
				tqdm.write("Assuming deeplake dataset with no labels")
				HAS_LABELS=False
		if HAS_LABELS:
			images, labels = data
		else:
			images = data['images']
			labels = torch.zeros((images.shape[0],)).to(torch.int64)

		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		normalized_images = images.float() / (cfg.color_levels - 1)
		if saveflag:
			saveflag = False
			save_samples(normalized_images,"samples","train_images.png")

		# *_, img_c, img_h, img_w = images.shape
		# mask = torch.ones((img_c, img_h, img_w), dtype=torch.float).to(device)
		# outputs = model(normalized_images, labels)
		# loss = F.cross_entropy(outputs, images, reduction="none")
		# masked_loss = loss * mask.unsqueeze(0).unsqueeze(0)
		# torch.mean(masked_loss).backward()

		optimizer.zero_grad()
		outputs = model(normalized_images, labels)
		loss = F.cross_entropy(outputs, images)
		loss.backward()
		if not cfg.overfit:
			clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)

		optimizer.step()
	scheduler.step()


def test_and_sample(cfg, model, device, test_loader, height, width, losses, params, epoch):
	test_loss = 0

	model.eval()
	HAS_LABELS = None
	saveflag=cfg.save_sample # turn on to save a batch
	with torch.no_grad():
		for data in tqdm(test_loader, desc="Testing".ljust(20)):
			if HAS_LABELS is None:
				HAS_LABELS=True
				try:
					images, labels = data
				except ValueError:
					print("Assuming deeplake dataset with no labels")
					HAS_LABELS=False
			if HAS_LABELS:
				images, labels = data
			else:
				images = data['images']
				labels = torch.zeros((images.shape[0],)).to(torch.int64)
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			normalized_images = images.float() / (cfg.color_levels - 1)
			if saveflag:
				saveflag = False
				save_samples(normalized_images,"samples","test_images.png")
			outputs = model(normalized_images, labels)

			test_loss += F.cross_entropy(outputs, images, reduction='none')

	test_loss = test_loss.mean().cpu() / len(test_loader.dataset)

	wandb.log({
		"Test loss": test_loss
	})
	print("Average test loss: {}".format(test_loss))

	losses.append(test_loss)
	params.append(model.state_dict())

	if cfg.dataset == "mnist":
		samples = model.sample((3, height, width), cfg.epoch_samples, label=None, device=device)
	else:
		samples = model.sample((3, height, width), cfg.epoch_samples, device=device)
	
	save_samples(samples, TRAIN_SAMPLES_DIR, 'epoch{}_samples.png'.format(epoch + 1))

def main():
	run = wandb.init(project="PixelCNN")
	wandb.config.update(cfg)
	torch.manual_seed(42)

	EPOCHS = cfg.epochs
	MODEL_PARAMS_OUTPUT_FILENAME = '{}_cks{}hks{}cl{}hfm{}ohfm{}hl{}_params.pth'\
		.format(cfg.dataset, cfg.causal_ksize, cfg.hidden_ksize, cfg.color_levels, cfg.hidden_fmaps, cfg.out_hidden_fmaps, cfg.hidden_layers)

	epoch_offset = 0
	if cfg.use_artifact:
		# artifact = run.use_artifact(cfg.use_artifact,type='model')
		# artifact_dir = os.path.join(artifact.download(),MODEL_PARAMS_OUTPUT_FILENAME)
		# model.load_state_dict(torch.load(artifact_dir))
		model, artifact_cfg, data = loadArtifactModel(run, cfg.use_artifact)
		print("Loaded artifact from",cfg.use_artifact)
		epoch_offset = data['epoch']
	else:
		model = PixelCNN(cfg=cfg)

	device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
	model.to(device)

	# train_loader, test_loader, HEIGHT, WIDTH = get_loaders(cfg.dataset, cfg.batch_size, cfg.color_levels, TRAIN_DATASET_ROOT, TEST_DATASET_ROOT)
	train_loader, test_loader, HEIGHT, WIDTH = get_loaders(cfg, TRAIN_DATASET_ROOT, TEST_DATASET_ROOT)
	# for b in test_loader:
	#     print(b[0].size())
	#     exit()

	# if cfg.overfit:
	#     cfg.learning_rate = .9
	#     cfg.weight_decay=0
	#     cfg.max_norm=9e9
	optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	scheduler = optim.lr_scheduler.CyclicLR(optimizer, cfg.learning_rate, 10*cfg.learning_rate, cycle_momentum=False)

	wandb.watch(model)

	losses = []
	params = []

	# samples = model.sample((3, HEIGHT, WIDTH), cfg.epoch_samples, device=device)
	# save_samples(samples, TRAIN_SAMPLES_DIR, 'epoch{}_samples.png'.format(0 + 1))
	cfg.epochs+=epoch_offset
	for epoch in range(EPOCHS):
		epoch+=epoch_offset
		if cfg.overfit:
			for _ in range(100):
				train(cfg, model, device, train_loader, optimizer, scheduler, epoch)
		train(cfg, model, device, train_loader, optimizer, scheduler, epoch)
		saveModel(None, model, cfg, "model/train_epoch_{}".format(epoch+1),data={"epoch":epoch+1})
		test_and_sample(cfg, model, device, test_loader, HEIGHT, WIDTH, losses, params, epoch)
		saveModel(run, model, cfg, "model/test_epoch_{}".format(epoch+1),data={"epoch":epoch+1, "loss":losses[-1].item()})

	print('\nBest test loss: {}'.format(np.amin(np.array(losses))))
	print('Best epoch: {}'.format(np.argmin(np.array(losses)) + 1))
	best_params = params[np.argmin(np.array(losses))]

	if not os.path.exists(MODEL_PARAMS_OUTPUT_DIR):
		os.mkdir(MODEL_PARAMS_OUTPUT_DIR)
	torch.save(best_params, os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME))
	artifact = wandb.Artifact("trained_model", type='model')
	artifact.add_file(os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME))
	run.log_artifact(artifact)
	run.finish()

if __name__ == '__main__':
	main()
