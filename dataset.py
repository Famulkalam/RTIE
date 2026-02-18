"""
RTIE — PyTorch Dataset and DataLoader

Loads synthetic thermal images with physics features, labels, and efficiency scores.
Applies train/val/test split (70/15/15) with augmentation on train set.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import config


class RadiatorThermalDataset(Dataset):
    """Dataset for radiator thermal fault detection."""

    def __init__(self, root_dir=config.SYNTHETIC_DIR, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load image
        img_path = os.path.join(self.root_dir, row["filename"])
        image = Image.open(img_path).convert("L")  # grayscale

        # Replicate to 3 channels for EfficientNet
        image = Image.merge("RGB", [image, image, image])

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Load physics features
        class_name = row["class"]
        idx_num = int(row["filename"].split("/")[-1].replace(".png", ""))
        feat_path = os.path.join(self.root_dir, class_name, f"{idx_num:04d}_features.npy")
        physics_features = torch.tensor(np.load(feat_path), dtype=torch.float32)

        # Labels
        label = torch.tensor(row["class_idx"], dtype=torch.long)
        efficiency = torch.tensor(row["efficiency_score"], dtype=torch.float32)

        return image, physics_features, label, efficiency


class GaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_transforms(train=True):
    """Get image transforms for train/val."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            # Gaussian Noise Augmentation (p=0.5)
            # Mixes clean and noisy data to ensure performance on both
            transforms.RandomApply([GaussianNoise(std=0.05)], p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=2):
    """Create train/val/test dataloaders with stratified split."""
    # Full datasets with different transforms
    full_dataset_train = RadiatorThermalDataset(transform=get_transforms(train=True))
    full_dataset_eval = RadiatorThermalDataset(transform=get_transforms(train=False))

    labels = full_dataset_train.metadata["class_idx"].values
    indices = np.arange(len(full_dataset_train))

    # Stratified split: 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        indices, test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=labels, random_state=42
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        stratify=temp_labels, random_state=42
    )

    train_set = Subset(full_dataset_train, train_idx)
    val_set = Subset(full_dataset_eval, val_idx)
    test_set = Subset(full_dataset_eval, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset splits:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    images, physics, labels, efficiencies = batch
    print(f"\nBatch shapes:")
    print(f"  Images:      {images.shape}")
    print(f"  Physics:     {physics.shape}")
    print(f"  Labels:      {labels.shape}")
    print(f"  Efficiencies:{efficiencies.shape}")
