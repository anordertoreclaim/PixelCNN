import numpy as np
import argparse
import os
import pickle
import os
from pixelcnn import PixelCNN

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision import datasets, transforms


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

# class LabeledDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         data = self.dataset[index]['images']
#         return data, 0  # Adding label 0 to each data sample

def labeled_collate_fn(batch):
    data = [b['images'] for b in batch]
    labels = torch.zeros(len(batch), dtype=torch.int32)
    return torch.stack(data), labels

def get_loaders(cfg, train_root, test_root):
    dataset_name = cfg.dataset
    batch_size = cfg.batch_size
    color_levels = cfg.color_levels

    normalize = transforms.Lambda(lambda image: np.array(image) / 255)

    discretize = transforms.Compose([
        transforms.Lambda(lambda image: quantisize(image, color_levels)),
        transforms.ToTensor()
    ])

    to_rgb = transforms.Compose([
        discretize,
        transforms.Lambda(lambda image_tensor: image_tensor.repeat(3, 1, 1))
    ])


    dataset_mappings = {'mnist': 'MNIST', 'fashionmnist': 'FashionMNIST',
     'cifar': 'CIFAR10', 'celeba':'CelebA', 'celeba-faces':'CelebA-Faces'}
    hw_mappings = {'mnist': (28, 28), 'fashionmnist': (28, 28), 'cifar': (32, 32), 'celeba': (32,32)}
    transform_mappings = {'mnist': to_rgb, 'fashionmnist': to_rgb, 'cifar': transforms.Compose([normalize, discretize]),
         'celeba':transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(hw_mappings['celeba']),
            normalize,
            discretize
            ])}

    try:
        dataset = dataset_mappings[dataset_name]
        transform = transform_mappings[dataset_name]

        if dataset == "CelebA":
            # dataset_path = ".data/celeba/"
            # print("Loading dataset",dataset_path)
            # train_dataset = getattr(datasets, dataset)(root=dataset_path, split="train", download=True, transform=transform)
            # test_dataset = getattr(datasets, dataset)(root=dataset_path, split="valid", download=True, transform=transform)
            import deeplake
            deeplake_kwargs = {
                "num_workers":2,
                "batch_size":batch_size,
                "pin_memory":True,
                "use_local_cache":True,
                "transform":{"images": transform},
                "shuffle":False,
                "decode_method":{"images":"pil"},
                "collate_fn":labeled_collate_fn,
            }
            ds_kwargs = {
                "read_only":True,
                "check_integrity":False,
            }
            # train_loader = deeplake.load("hub://activeloop/celeb-a-train").pytorch(**deeplake_kwargs)
            # test_loader = deeplake.load("hub://activeloop/celeb-a-test").pytorch(**deeplake_kwargs)
            train_dataset = deeplake.load("hub://activeloop/celeb-a-train", **ds_kwargs)
            test_dataset = deeplake.load("hub://activeloop/celeb-a-test", **ds_kwargs)
        elif dataset == "CelebA-Faces":
            # TODO: Use hugginface repo for datasets
            from datasets import load_dataset
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ds=load_dataset("nielsr/CelebA-faces",split="train").with_format("torch",device=device)
        else:
            train_dataset = getattr(datasets, dataset)(root=train_root, train=True, download=True, transform=transform)
            test_dataset = getattr(datasets, dataset)(root=test_root, train=False, download=True, transform=transform)

        h, w = hw_mappings[dataset_name]
    except KeyError:
        raise AttributeError("Unsupported dataset")

    if cfg.dataset_size is not None:
        if dataset != "CelebA":
            train_dataset = Subset(train_dataset, list(range(cfg.dataset_size)))
            test_dataset = Subset(test_dataset, list(range(cfg.dataset_size)))
        else:
            train_dataset, _ = train_dataset.random_split([cfg.dataset_size, len(train_dataset)-cfg.dataset_size])
            test_dataset, _ = test_dataset.random_split([cfg.dataset_size, len(test_dataset)-cfg.dataset_size])
    if dataset != "CelebA":
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    else:
        train_loader = train_dataset.pytorch(**deeplake_kwargs)
        test_loader = test_dataset.pytorch(**deeplake_kwargs)
    
    if cfg.overfit:
        test_loader = train_loader

    return train_loader, test_loader, h, w

def delete_contents(directory_path):
    assert "model/train" in directory_path, "Careful, trying to delete a non model/train folder"
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                # print("remove",item_path)
        print(f"Contents of directory '{directory_path}' have been successfully deleted.")
    except Exception as e:
        print(f"Error occurred while deleting contents of directory '{directory_path}': {e}")


def saveModel(run, model, cfg, path, data=None):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        delete_contents(path)
    if data is not None:
        with open(os.path.join(path,"data.pkl"), 'wb') as f:
            pickle.dump(data, f)
    with open(os.path.join(path,"cfg.pkl"), 'wb') as f:
        pickle.dump(cfg, f)
    torch.save(model.state_dict(), os.path.join(path, "model_state_dict.pth"))
    if run is not None and cfg.upload:
        artifact = wandb.Artifact(f"{cfg.dataset}_model", type='model')
        artifact.add_dir(local_path=path)
        run.log_artifact(artifact)
        print("Saved model to wandb")
    else:
        print("Saved model locally to",path)

def loadArtifactModel(run, artifactName):
    artifact = run.use_artifact(artifactName)
    path = artifact.download()
    with open(os.path.join(path, "cfg.pkl"), 'rb') as f:
        cfg = pickle.load(f)
    model = PixelCNN(cfg=cfg)  # Instantiate your model class here
    model.load_state_dict(torch.load(os.path.join(path, "model_state_dict.pth")))

    data = None
    data_path = os.path.join(path, "data.pkl")
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    
    return model, cfg, data