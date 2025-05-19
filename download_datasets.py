#!/usr/bin/env python3
"""
download_datasets.py

Download MNIST, CIFAR‑10, CIFAR‑100 and Tiny‑ImageNet into a local folder.
"""
import argparse, os, shutil, urllib.request, zipfile

import torchvision
import torchvision.transforms as T

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data", help="Target directory")
    return p.parse_args()

def get_dataset(dataset_cls, root, **kwargs):
    _ = dataset_cls(root=root, download=True, **kwargs)

def download_tiny(root):
    dst = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(dst):
        print("Tiny‑ImageNet already present – skipping …")
        return
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    print("▶ Downloading Tiny‑ImageNet…")
    urllib.request.urlretrieve(TINY_URL, zip_path)
    print("▶ Extracting …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    os.remove(zip_path)
    # --- Reorganize validation images into class subfolders ---
    val_dir = os.path.join(dst, "val")
    images_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")
    print("▶ Reorganizing validation images…")
    with open(annot_file) as f:
        for line in f:
            img, cls, *_ = line.split('\t')
            cls_folder = os.path.join(images_dir, cls)
            os.makedirs(cls_folder, exist_ok=True)
            shutil.move(os.path.join(images_dir, img),
                        os.path.join(cls_folder, img))
    print("✓ Tiny‑ImageNet ready at", dst)

def main():
    args = parse_args()
    root = args.data_root
    os.makedirs(root, exist_ok=True)

    print("▶ MNIST …")
    get_dataset(torchvision.datasets.MNIST, root, train=True, transform=T.ToTensor())
    get_dataset(torchvision.datasets.MNIST, root, train=False, transform=T.ToTensor())

    print("▶ CIFAR‑10 …")
    get_dataset(torchvision.datasets.CIFAR10, root, train=True, transform=T.ToTensor())
    get_dataset(torchvision.datasets.CIFAR10, root, train=False, transform=T.ToTensor())

    print("▶ CIFAR‑100 …")
    get_dataset(torchvision.datasets.CIFAR100, root, train=True, transform=T.ToTensor())
    get_dataset(torchvision.datasets.CIFAR100, root, train=False, transform=T.ToTensor())

    print("▶ Tiny‑ImageNet …")
    try:
        download_tiny(root)
    except Exception as exc:
        print("⚠ Tiny‑ImageNet download/reorg failed:", exc)
        print("  Please download manually and unzip to", root)

#!/usr/bin/env python3
import os
import shutil

def reorganize_tiny_val(root="data/tiny-imagenet-200"):
    val_dir = os.path.join(root, "val")
    images_dir = os.path.join(val_dir, "images")
    anno_path = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(images_dir) or not os.path.isfile(anno_path):
        raise FileNotFoundError("val/images or val_annotations.txt not found.")

    print(f"🛠️ Reorganizing TinyImageNet validation set at {val_dir}")

    # Make class folders and move images into them
    with open(anno_path) as f:
        for line in f:
            img, cls = line.strip().split("\t")[:2]
            cls_dir = os.path.join(images_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_dir, img)
            dst = os.path.join(cls_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)

    print("✅ Validation images moved into class folders.")

if __name__=="__main__":
    main()
    reorganize_tiny_val()
