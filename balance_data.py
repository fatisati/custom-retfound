import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np

def get_balanced_sampler(dataset, more_augmentation=1):
    """
    Creates a PyTorch DataLoader with balanced oversampling and suitable augmentations for retina OCT images.

    Args:
        dataset_dir (str): Path to the dataset folder (organized in subfolders by class).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader with oversampling and augmentation applied.
    """
    if more_augmentation == 1:
        # Define data augmentations suitable for retina OCT images
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Slight rotation
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Random cropping
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Slight color variations
        ])
        # Combine the dataset's existing transform with augmentations
        if dataset.transform:
            dataset.transform = transforms.Compose([augmentations, dataset.transform])
        else:
            dataset.transform = augmentations
        print('using new balanced sampler with augmentaion')
    
    # Get class counts and compute weights
    class_counts = np.bincount([label for _, label in dataset.samples])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in dataset.samples]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler
