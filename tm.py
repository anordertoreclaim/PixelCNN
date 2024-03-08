import wandb
run = wandb.init(project="PixelCNN")

artifact = wandb.Artifact("celeba_model", type='model')
artifact.add_dir(local_path="model/test_epoch_1")
run.log_artifact(artifact)