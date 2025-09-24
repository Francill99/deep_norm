#!/usr/bin/env python3
"""

Example:
    python train_model.py --device cuda:0 --P 5000 --T 1000 --lr 1e-3 --dataset CIFAR10
"""
import argparse
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from deep_norm.cnn.model import CNNClassifier
from deep_norm.train.training import train



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0", help="PyTorch device string (e.g. 'cpu', 'cuda:0')")
    parser.add_argument("--P", type=int, required=True, help="Number of training samples to keep (rest used for validation)")
    parser.add_argument("--T", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning‑rate for Adam")
    parser.add_argument("--seed", type=int, default=5, help="Seed")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10", "CIFAR100", "TINYIMAGENET"], required=True,
                        help="Which dataset to use")
    parser.add_argument("--wdecay", type=float, default=0., help="Weight decay")
    parser.add_argument("--init_factor", type=float, default=1.0, help="Factor to multiply all weights at initialization")
    parser.add_argument("--data_root", default="./data", help="Root folder **already** containing the datasets")
    parser.add_argument("--out_dir", default="./savings", help="Where to store the numpy log file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if name == "TINYIMAGENET":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    raise ValueError(f"Unknown dataset '{name}'.")


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
        base = os.path.join(root, "tiny-imagenet-200")
        train_dir = os.path.join(base, "train")
        val_images = os.path.join(base, "val", "images")
        train_ds = ImageFolder(train_dir, transform=transform)
        test_ds  = ImageFolder(val_images, transform=transform)
        return train_ds, test_ds, (3, 64, 64), 200


    raise ValueError

def compute_random_margins(model: nn.Module, dataset, P_margins: int, device: str,
                           batch_size: int = 256) -> np.ndarray:
    P_margins = min(P_margins, len(dataset))
    idx = torch.randperm(len(dataset))[:P_margins]
    subset = Subset(dataset, idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    margins: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            m = model.compute_margin_distribution(x, y)
            margins.append(m.cpu())
    return torch.cat(margins, dim=0).numpy()



def build_model(input_size, num_classes, device, dataset, initialization_factor):
    """
    Dataset-specific CNNs tuned for best offline baselines:
      • CIFAR-10: 3 conv-blocks with 32→64→128 filters, moderate dropout
      • CIFAR-100: 4 conv-blocks with 64→128→256→512 filters, stronger dropout
      • TinyImageNet: 4 conv-blocks (32→32→64→64→128), heavy FC head + high dropout
    """
    c_in = input_size[0]

    if dataset == "CIFAR10" or dataset== "MNIST":
        # Two conv layers per pooling block; pool every 2 convs
        model = CNNClassifier(
            conv_channels=[c_in, 32, 64, 128],    # 3 blocks: 32→64→128 filters 
            kernel_size=3,
            mlp_layers=[512, num_classes],        # 512-dim hidden → 10 classes
            pool_every=2,                         # pool after every 2 convs 
            dropout=0.,                          # 30% dropout (best at 20–40%) 
            input_size=input_size,
            initialization_factor=initialization_factor,
        ).to(device)

    elif dataset == "CIFAR100":
        # Four conv blocks, doubling channels each time
        model = CNNClassifier(
            conv_channels=[c_in, 64, 128, 256, 512],  # 4 blocks 
            kernel_size=3,
            mlp_layers=[1024, 512, num_classes],     # wider FC
            pool_every=2,                            # pool every two conv layers 
            dropout=0.1,                             # stronger dropout for 100 classes 
            input_size=input_size,
            initialization_factor=initialization_factor
        ).to(device)

    elif dataset == "TINYIMAGENET":
        # TinyImageNet baseline
        model = CNNClassifier(
            conv_channels=[c_in, 32, 32, 64, 64, 128],  # M4: 32→32→64→64→128 
            kernel_size=3,
            mlp_layers=[2048, 1024, 512, num_classes],  # deep head for 200-way
            pool_every=2,                              # as in M4 architecture 
            dropout=0.2,                               # 50% dropout on FC layers 
            input_size=input_size,
            initialization_factor=initialization_factor
        ).to(device)

    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    return model


def main():
    args = parse_args()
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")

    set_seed(args.seed)

    transform = get_transforms(args.dataset)
    train_ds, test_ds, input_size, num_classes = load_datasets(args.dataset, args.data_root, transform)

    # Train/validation split
    p = min(args.P, len(train_ds))
    train_size, val_size = p, len(train_ds) - p
    train_set, val_set = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    model = build_model(input_size, num_classes, device, args.dataset, args.init_factor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    criterion = nn.CrossEntropyLoss()

    logs = train(model, train_loader, val_loader, test_loader, device, optimizer, criterion, args.P, args.T, other_norms=True, norms_every_steps=None)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name_file_save = f"CNN_{args.dataset}_P{args.P}_seed{args.seed}_WD{args.wdecay}_INIT{args.init_factor}.npy"
    np.save(Path(args.out_dir) / name_file_save , logs)
    print("Training complete. Logs saved to", Path(args.out_dir) / name_file_save)


if __name__ == "__main__":
    main()
