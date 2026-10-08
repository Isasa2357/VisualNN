
import shutil
import os
import subprocess
from enum import Enum

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100

class Dataset(Enum):
    IMAGENETTE160 = 1
    IMAGENETTE320 = 2
    CIFAR10 = 3
    CIFAR100 = 4

def get_dataset_enum(dataset: str) -> Dataset:
    if dataset.lower() == "imagenette2-160":
        return Dataset.IMAGENETTE160
    elif dataset.lower() == "imagenette2-320":
        return Dataset.IMAGENETTE320
    elif dataset.lower() == "cifar10":
        return Dataset.CIFAR10
    elif dataset.lower() == "cifar100":
        return Dataset.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_dataset_str(dataset: Dataset) -> str:
    if dataset == Dataset.IMAGENETTE160:
        return "imagenette2-160"
    elif dataset == Dataset.IMAGENETTE320:
        return "imagenette2-320"
    elif dataset == Dataset.CIFAR10:
        return "cifar10"
    elif dataset == Dataset.CIFAR100:
        return "cifar100"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_class_num(dataset: Dataset) -> int:
    if dataset == Dataset.IMAGENETTE160:
        return 10
    elif dataset == Dataset.IMAGENETTE320:
        return 10
    elif dataset == Dataset.CIFAR10:
        return 10
    elif dataset == Dataset.CIFAR100:
        return 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_dataset_path(dataset: Dataset) -> tuple[str, str, str]:
    strage_root: str = "Z:\\dataset\\VisionClassification"
    if dataset == Dataset.IMAGENETTE160:
        return f"{strage_root}\\imagenette2-160", f"{strage_root}\\imagenette2-160\\train", f"{strage_root}\\imagenette2-160\\val"
    elif dataset == Dataset.IMAGENETTE320:
        return f"{strage_root}\\imagenette2-320", f"{strage_root}\\imagenette2-320\\train", f"{strage_root}\\imagenette2-320\\val"
    elif dataset == Dataset.CIFAR10:
        return f"{strage_root}\\cifar10", f"{strage_root}\\cifar10\\train", f"{strage_root}\\cifar10\\val"
    elif dataset == Dataset.CIFAR100:
        return f"{strage_root}\\cifar100", f"{strage_root}\\cifar100\\train", f"{strage_root}\\cifar100\\val"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def make_dataloader(dataset: Dataset, batch_size: int, train_transform: transforms.Compose, val_transform: transforms.Compose, download: bool = False) -> tuple[DataLoader, DataLoader]:

    root_path, train_path, val_path = get_dataset_path(dataset)

    # ダウンロードがTrueなら，実行ディレクトリにデータセットをコピー
    if download:
        dataset_str = get_dataset_str(dataset)
        subprocess.run(["robocopy", root_path, f"./dataset/{dataset_str}", "/E", "/MT:8"])

        # コピー後にtrain_pathとval_pathを更新
        train_path = f"./dataset/{dataset_str}/train"
        val_path = f"./dataset/{dataset_str}/val"


    if dataset == Dataset.IMAGENETTE160:
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_path, transform=val_transform)
    elif dataset == Dataset.IMAGENETTE320:
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_path, transform=val_transform)
    elif dataset == Dataset.CIFAR10:
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_path, transform=val_transform)
    elif dataset == Dataset.CIFAR100:
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_path, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader