#!/usr/bin/env python3
"""
train_vit.py

Train a lightweight Vision‑Transformer (ViT) on CIFAR‑10, CIFAR‑100
or Tiny‑ImageNet using only P training samples for T epochs.

Example:
    python train_vit.py --device cuda:0 --P 5000 --T 600 --lr 3e-4 --dataset CIFAR10
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

from deep_norm.vit.model import ViT                 
from deep_norm.train.training import train

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--P",  type=int, required=True, help="# training samples to keep")
    p.add_argument("--T",  type=int, default=600,   help="# epochs")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning‑rate")
    p.add_argument("--seed",   type=int, default=5)
    p.add_argument("--dataset", choices=["MNIST","CIFAR10", "CIFAR100", "TINYIMAGENET"],
                   required=True)
    p.add_argument("--wdecay",  type=float, default=0.)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir",   default="./savings")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_datasets(name: str, root: str, transform):
    """
    Return (train_ds, test_ds, input_size_tuple, num_classes)
    ------------------------------------------------------------------
    • MNIST        → (1, 28, 28), 10
    • CIFAR‑10/100 → (3, 32, 32), 10 / 100
    • Tiny‑ImageNet→ (3, 64, 64), 200
    """
    root = Path(root)

    if name == "MNIST":
        tr = torchvision.datasets.MNIST(
            root, train=True,  download=False, transform=transform
        )
        te = torchvision.datasets.MNIST(
            root, train=False, download=False, transform=transform
        )
        return tr, te, (1, 28, 28), 10

    if name == "CIFAR10":
        tr = torchvision.datasets.CIFAR10(
            root, train=True,  download=False, transform=transform
        )
        te = torchvision.datasets.CIFAR10(
            root, train=False, download=False, transform=transform
        )
        return tr, te, (3, 32, 32), 10

    if name == "CIFAR100":
        tr = torchvision.datasets.CIFAR100(
            root, train=True,  download=False, transform=transform
        )
        te = torchvision.datasets.CIFAR100(
            root, train=False, download=False, transform=transform
        )
        return tr, te, (3, 32, 32), 100
    
    if name == "TINYIMAGENET":
        base  = os.path.join(root, "tiny-imagenet-200")
        trdir = os.path.join(base, "train")
        valim = os.path.join(base, "val", "images")
        tr = ImageFolder(trdir, transform=transform)
        te = ImageFolder(valim, transform=transform)
        return tr, te, (3, 64, 64), 200


    raise ValueError(f"Unknown dataset '{name}'.")

def get_transforms(name: str):
    """
    Return a torchvision.transforms.Compose for each dataset:
      • MNIST        → RandomAffine + ToTensor + Normalize(0.1307,0.3081)
      • CIFAR-10/100 → ToTensor + Normalize(0.4914…,0.2023…)
      • Tiny-ImageNet→ Resize(64) + ToTensor + Normalize(0.4802…,0.2770…)
    """
    if name == "MNIST":
        return transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    if name in {"CIFAR10", "CIFAR100"}:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if name == "TINYIMAGENET":
        mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    raise ValueError(f"Unknown dataset '{name}'.")

def build_model(input_size, num_classes, device, dataset):
    """
    Hyper‑parameters chosen to keep the model count <≈ 10 M while maintaining
    strong accuracy; roughly ViT‑Tiny for CIFAR‑10, a mid‑tiny for CIFAR‑100,
    and a small for Tiny‑ImageNet.
    """
    c, H, _ = input_size

    if dataset == "MNIST":
        model = ViT(
            image_size = H,    
            patch_size = 7,    
            in_chans   = c,    
            embed_dim  = 128,
            depth      = 4,
            num_heads  = 4,
            mlp_ratio  = 4,
            dropout    = 0.,
            num_classes= num_classes
        ).to(device)
        
    elif dataset == "CIFAR10":
        
        model = ViT(image_size=H, patch_size=4, in_chans=c,
                    embed_dim=192, depth=6, num_heads=3,
                    mlp_ratio=4, dropout=0.,
                    num_classes=num_classes).to(device)

    elif dataset == "CIFAR100":
   
        model = ViT(image_size=H, patch_size=4, in_chans=c,
                    embed_dim=256, depth=6, num_heads=4,
                    mlp_ratio=4, dropout=0.,
                    num_classes=num_classes).to(device)

    elif dataset == "TINYIMAGENET":
 
        model = ViT(image_size=H, patch_size=4, in_chans=c,
                    embed_dim=384, depth=6, num_heads=6,
                    mlp_ratio=4, dropout=0.,
                    num_classes=num_classes).to(device)
    else:
        raise ValueError(dataset)
    return model


def main():
    args   = parse_args()
    device = torch.device(args.device if ("cuda" in args.device and
                                          torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    transform = get_transforms(args.dataset)
    tr_ds, te_ds, input_size, num_classes = load_datasets(
        args.dataset, args.data_root, transform)


    P = min(args.P, len(tr_ds))
    tr_size, val_size = P, len(tr_ds) - P
    train_set, val_set = random_split(tr_ds, [tr_size, val_size])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, num_workers=4)
    test_loader  = DataLoader(te_ds,     batch_size=128, shuffle=False, num_workers=4)

    model     = build_model(input_size, num_classes, device, args.dataset)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    loss_fn   = nn.CrossEntropyLoss()

    logs = train(model, train_loader, val_loader, test_loader,
                 device, optimizer, loss_fn, args.P, args.T)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.out_dir) / f"ViT_{args.dataset}_P{args.P}_seed{args.seed}_WD{args.wdecay}.npy",
            logs)
    print("Finished training. Logs saved.")

if __name__ == "__main__":
    main()
