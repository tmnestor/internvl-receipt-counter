"""
Dataset implementations for receipt counting with InternVL2.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ReceiptDataset(Dataset):
    """
    Dataset for receipt counting with InternVL2 support.
    
    Implements efficient loading and preprocessing with torchvision transforms.
    """
    def __init__(
        self,
        csv_file: Union[str, Path],
        img_dir: Union[str, Path],
        transform: Optional[T.Compose] = None,
        augment: bool = False,
        binary: bool = False,
        max_samples: Optional[int] = None,
        image_size: int = 448,
    ):
        """
        Initialize a receipt dataset.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            image_size: Target image size (448 for InternVL2)
            augment: Whether to apply data augmentation (used for training)
            binary: Whether to use binary classification mode (0 vs 1+ receipts)
            max_samples: Optional limit on number of samples (useful for quick testing)
        """
        self.data = pd.read_csv(csv_file)
        if max_samples is not None:
            self.data = self.data.sample(min(len(self.data), max_samples))
            
        self.img_dir = Path(img_dir)
        self.binary = binary
        self.image_size = image_size
        
        # Set up transforms
        if transform:
            self.transform = transform
        else:
            # ImageNet normalization values
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Default transformations based on model requirements
            if augment:
                self.transform = T.Compose([
                    T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.1),  # Occasionally flip vertically (some receipts are upside down)
                    T.RandomAffine(
                        degrees=30,
                        translate=(0.1, 0.1),
                        scale=(0.8, 1.2),
                    ),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    normalize,
                ])
            else:
                # Validation/test transforms
                self.transform = T.Compose([
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                    normalize,
                ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.data.iloc[idx, 0]
        image_path = self.img_dir / filename
        
        try:
            # Read image using PIL
            image = Image.open(image_path).convert('RGB')
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to a blank image
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Get receipt count
        count = int(self.data.iloc[idx, 1])
        
        # Convert to appropriate classification label
        if self.binary:
            # Binary: 0 vs 1+ receipts
            label = 1 if count > 0 else 0
        else:
            # Default: 3-class (0, 1, 2+ receipts)
            label = min(count, 2)  # Map all counts ≥2 to class 2
            
        return image_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(config) -> Dict[str, DataLoader]:
    """
    Create training, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = ReceiptDataset(
        config["data"]["train_csv"],
        config["data"]["train_dir"],
        augment=config["data"]["augmentation"],
        binary=config["model"]["num_classes"] == 2,
        image_size=config["data"]["image_size"],
    )
    
    val_dataset = ReceiptDataset(
        config["data"]["val_csv"],
        config["data"]["val_dir"],
        augment=False,
        binary=config["model"]["num_classes"] == 2,
        image_size=config["data"]["image_size"],
    )
    
    test_dataset = None
    if "test_csv" in config["data"] and "test_dir" in config["data"]:
        test_dataset = ReceiptDataset(
            config["data"]["test_csv"],
            config["data"]["test_dir"],
            augment=False,
            binary=config["model"]["num_classes"] == 2,
            image_size=config["data"]["image_size"],
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
    }
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        loaders['test'] = test_loader
    
    return loaders