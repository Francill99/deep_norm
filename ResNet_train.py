#!/usr/bin/env python3
"""
train_resnet.py

Train a ResNetClassifier (from deep_norm) on CIFAR‑10, CIFAR‑100, or Tiny‑ImageNet
using precisely tuned hyper‑parameters for each dataset.

Before running, make sure you have executed *download_datasets.py* so the datasets
are cached locally – this avoids network fetches on subsequent runs.

Example:
    python train_resnet.py --device cuda:0 --P 20000 --T 400 --dataset CIFAR100
"""
import argparse
import os
from pathlib import Path
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# Deep‑norm imports – adjust the dotted path to match your project layout
# -----------------------------------------------------------------------------
from deep_norm.resnet.model import ResNetClassifier, BasicBlock  # noqa: E402
from deep_norm.train.training import train  # noqa: E402

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ResNet training script with dataset‑specific hyper‑parameter presets.",
    )
    p.add_argument("--device", default="cuda:0", help="PyTorch device string (e.g. 'cpu', 'cuda:0')")
    p.add_argument("--P", type=int, required=True, help="Number of training samples to keep (rest → validation)")
    p.add_argument("--T", type=int, default=None, help="Number of training epochs; default depends on dataset")
    p.add_argument("--lr", type=float, default=None, help="Learning rate; default depends on dataset")
    p.add_argument("--seed", type=int, default=5, help="Seed")
    p.add_argument(
        "--dataset",
        choices=["MNIST","CIFAR10", "CIFAR100", "TINYIMAGENET"],
        required=True,
        help="Dataset to use",
    )
    p.add_argument("--wdecay", type=float, default=0., help="Weight decay")
    p.add_argument("--data_root", default="./data", help="Root folder **already** containing the datasets")
    p.add_argument("--out_dir", default="./savings", help="Where to store the numpy log file")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def get_transforms(name: str):
    if name == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    if name in {"CIFAR10", "CIFAR100"}:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if name == "TINYIMAGENET":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
        return transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    raise ValueError(f"Unknown dataset '{name}'.")


def _load_tiny_imagenet(root: str, transform):
    base = os.path.join(root, "tiny-imagenet-200")
    train_dir = os.path.join(base, "train")
    val_images = os.path.join(base, "val", "images")
    train_ds = ImageFolder(train_dir, transform=transform)
    test_ds = ImageFolder(val_images, transform=transform)
    return train_ds, test_ds


def load_datasets(name: str, root: str, transform):
    """Return (train_ds, test_ds, input_size, num_classes)."""
    root = Path(root)
    if name == "MNIST":
        train_ds = torchvision.datasets.MNIST(root, train=True, download=False, transform=transform)
        test_ds = torchvision.datasets.MNIST(root, train=False, download=False, transform=transform)
        return train_ds, test_ds, (1, 28, 28), 10
    if name == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=transform)
        return train_ds, test_ds, (3, 32, 32), 10

    if name == "CIFAR100":
        train_ds = torchvision.datasets.CIFAR100(root, train=True, download=False, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=transform)
        return train_ds, test_ds, (3, 32, 32), 100

    if name == "TINYIMAGENET":
        train_ds, test_ds = _load_tiny_imagenet(root, transform)
        return train_ds, test_ds, (3, 64, 64), 200

    raise ValueError

# -----------------------------------------------------------------------------
# Hyper‑parameter presets
# -----------------------------------------------------------------------------

def dataset_presets(name: str):
    """Return (layers, widths, epochs, lr, batch_size, weight_decay, dropout)."""
    if name == "CIFAR10" or name== "MNIST":
        return ([2, 2, 2, 2], [64, 128, 256, 512], 350, 1e-3, 128, 5e-4, 0.0)
    if name == "CIFAR100":
        return ([3, 4, 6, 3], [64, 128, 256, 512], 400, 1e-3, 128, 5e-4, 0.0)
    if name == "TINYIMAGENET":
        return ([3, 4, 6, 3], [64, 128, 256, 512], 150, 3e-4, 256, 1e-4, 0.0)
    raise ValueError

# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def build_model(
    input_size: Tuple[int, int, int],
    num_classes: int,
    dataset: str,
    device: torch.device,
):
    layers, widths, _, _, batch_size, _, dropout = dataset_presets(dataset)
    model = ResNetClassifier(
        layers=layers,
        widths=widths,
        block=BasicBlock,
        num_classes=num_classes,
        dropout=dropout,
        input_size=input_size,
    ).to(device)
    return model

# -----------------------------------------------------------------------------
# Training logic
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    set_seed(args.seed)

    # fetch dataset‑specific presets
    layers, widths, _, _, batch_size, _, _ = dataset_presets(args.dataset)
    epochs = args.T
    lr = args.lr 

    transform = get_transforms(args.dataset)
    train_ds, test_ds, input_size, num_classes = load_datasets(args.dataset, args.data_root, transform)

    # Train/validation split
    p = min(args.P, len(train_ds))
    train_size, val_size = p, len(train_ds) - p
    train_set, val_set = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(input_size, num_classes, args.dataset, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.wdecay)
    criterion = nn.CrossEntropyLoss()

    logs = train(model, train_loader, val_loader, test_loader, device, optimizer, criterion, args.P, epochs)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.out_dir) / f"ResNet_{args.dataset}_P{args.P}_seed{args.seed}_WD{args.wdecay}.npy"
    np.save(save_path, logs)
    print("Training complete. Logs saved to", save_path)


if __name__ == "__main__":
    main()
