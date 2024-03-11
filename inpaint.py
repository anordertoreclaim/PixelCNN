import torch
import pickle
import deeplake
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
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
cfg.save_sample = True
def inpaint_test(cfg, model, device, test_loader):
	test_loss = 0


	saveflag = cfg.save_sample
	model.eval()
	with torch.no_grad():
		for images, labels in tqdm(test_loader, desc="Inpainting".ljust(20)):
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			mask = torch.zeros(images.shape[1:]).to(device,non_blocking=True)
			mask[:,5:25,5:25]=1
			mask[:,10:20,10:20]=0
			C,H,W =mask.shape
			mask = mask.unsqueeze(0).expand(images.shape[0],C,H,W)
			red_mask = torch.clone(mask)
			red_mask[:,1:,:,:]*=0

			normalized_images = images.float() / (cfg.color_levels - 1)
			if saveflag:
				saveflag = False
			outputs = model.inpaint(normalized_images*(1-mask),mask, labels)
			save_rows([normalized_images, normalized_images*(1-mask)+red_mask, outputs], "samples", "inpainted.png")
			# save_samples(normalized_images,"samples","ip_unmasked.png")
			# save_samples(normalized_images*(1-mask),"samples","ip_masked.png")
			# save_samples(outputs,"samples","ip_generated.png")
			break

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

	inpaint_test(cfg, model, device, test_loader)

	# if not os.path.exists(MODEL_PARAMS_OUTPUT_DIR):
	# 	os.mkdir(MODEL_PARAMS_OUTPUT_DIR)
	# torch.save(best_params, os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME))
	# artifact = wandb.Artifact("trained_model", type='model')
	# artifact.add_file(os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME))
	# run.log_artifact(artifact)
	run.finish()

if __name__ == '__main__':
	main()