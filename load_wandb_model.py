import torch
import os
import wandb
from args import getCFG

from utils import str2bool, save_samples
from pixelcnn import PixelCNN

TRAIN_SAMPLES_DIR = 'train_samples'

cfg = getCFG()

device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

torch.manual_seed(42)

run = wandb.init()
artifact = run.use_artifact('pixelcnn/PixelCNN/model:v2',type='model')
print(artifact)
artifact_dir = os.path.join(artifact.download(),"mnist_cks7hks7cl2hfm30ohfm10hl6_params.pth")
print("?",artifact_dir)

model.load_state_dict(torch.load(artifact_dir))
samples = model.sample((3, 28,28), cfg.epoch_samples, device=device, pbar=True)
save_samples(samples, TRAIN_SAMPLES_DIR, 'loaded_samples.png')