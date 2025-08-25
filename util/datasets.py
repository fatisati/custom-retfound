# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandomErasing
import random


def build_dataset(is_train, args):

    root = os.path.join(args.data_path, is_train)
    mean, std = args.mean, args.std

    if args.transform == "custom":
        transform = get_custom_transform(args, is_train)
    else:
        transform = build_transform(is_train, args, mean, std)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def calculate_mean_std(data_path, batch_size=16):
    """
    Calculate the mean and standard deviation of a dataset.

    Args:
        data_path (str): Path to the dataset directory.
        batch_size (int): Batch size for DataLoader. Default is 64.

    Returns:
        tuple: Mean and standard deviation as lists for each channel.
    """
    # Define a transform to convert images to tensors without normalization
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in tqdm(loader, desc="Calculating Mean and Std"):
        batch_images = images.size(0)
        images = images.view(batch_images, images.size(1), -1)  # Flatten HxW

        mean += images.mean(dim=2).sum(dim=0)  # Sum mean of each channel
        std += images.std(dim=2).sum(dim=0)  # Sum std of each channel
        total_images += batch_images

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()


def build_transform(is_train, args, mean, std):

    # train transform
    if is_train == "train":
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_custom_transform(args, is_train="train"):
    """
    Creates a suitable transformation pipeline for fine-tuning a transformer on OCT retina images.

    Args:
        args: Namespace containing necessary parameters like input_size, color_jitter, reprob, remode, recount, mean, std.
        is_train (str): The split type ('train', 'test', or 'val').

    Returns:
        A torchvision.transforms.Compose object containing the transformation pipeline.
    """
    # Shared transformations
    normalize = transforms.Normalize(mean=args.mean, std=args.std)
    resize = transforms.Resize(
        (args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC
    )

    if is_train == "train":
        # Training Transformations
        transform = transforms.Compose(
            [
                resize,
                transforms.ColorJitter(
                    brightness=args.color_jitter if args.color_jitter else 0.1,
                    contrast=args.color_jitter if args.color_jitter else 0.1,
                    saturation=args.color_jitter if args.color_jitter else 0.1,
                    hue=0.05,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))],
                    p=0.3,
                ),
                transforms.ToTensor(),
                normalize,
                RandomErasing(
                    p=args.reprob if hasattr(args, "reprob") else 0.25,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value="mean" if args.remode == "mean" else 0,
                    inplace=True,
                ),
            ]
        )
    elif is_train == "val":
        # Validation Transformations
        transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif is_train == "test":
        # Test Transformations
        transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(
            f"Unknown split type: {is_train}. Must be 'train', 'val', or 'test'."
        )

    return transform
