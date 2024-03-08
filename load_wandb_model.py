import torch
import os
import wandb
from args import getCFG

from utils import *
from pixelcnn import PixelCNN

TRAIN_SAMPLES_DIR = 'train_samples'

cfg = getCFG()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

if cfg.use_artifact:
	run = wandb.init()
	# artifact = run.use_artifact(cfg.use_artifact,type='model')
	# artifact_dir = os.path.join(artifact.download(),MODEL_PARAMS_OUTPUT_FILENAME)
	# model.load_state_dict(torch.load(artifact_dir))
	model, artifact_cfg, data = loadArtifactModel(run, cfg.use_artifact)
	print("Loaded artifact from",cfg.use_artifact)
else:
	print("ERROR: no artifact given")
	exit(1)

model.to(device)

samples = model.sample((3, 28,28), cfg.epoch_samples, device=device, pbar=True)
save_samples(samples, TRAIN_SAMPLES_DIR, 'loaded_samples.png')