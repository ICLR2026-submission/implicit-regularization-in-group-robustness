"""
Domino Dataset with CIFAR-10 and MNIST Spurious Correlation
"""

import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from collections import Counter, defaultdict

class SpuriousCorrelationDomino(torch.utils.data.Dataset):
    """
    Domino dataset combining CIFAR-10 and MNIST with spurious correlation.
    
    True labels come from CIFAR-10 images (top half).
    Spurious labels come from MNIST digits (bottom half).
    The correlation parameter controls alignment between CIFAR class and MNIST digit.
    
    Parameters
    ----------
    root : str
        Path to datasets.
    spurious_correlation : float, default=0.95
        Controls correlation between CIFAR-10 class and MNIST digit.
    selected_classes : list, optional
        Subset of classes to use (0-9). If None, uses all classes.
    """
    
    CLASS_NAMES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        spurious_correlation=0.95,
        selected_classes=None,
        mnist_selected_classes=None,
        seed=42,
        **kwargs,
    ):
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.spurious_correlation = spurious_correlation
        
        # Load both datasets
        self.cifar_dataset = CIFAR10(
            root=root, train=train, download=download
        )
        self.mnist_dataset = MNIST(
            root=root, train=train, download=download
        )
        
        # Handle class selection
        self.selected_classes = (
            list(range(10)) if selected_classes is None else selected_classes
        )
        self.n_classes = len(self.selected_classes)
        assert 0 < self.n_classes <= 10
        assert all(c in range(10) for c in self.selected_classes)
        self.mnist_selected_classes = mnist_selected_classes if mnist_selected_classes is not None else self.selected_classes
        
        # Create class mapping
        self.class2idx = {c: i for i, c in enumerate(self.selected_classes)}
        self.mnist_class2idx = {c: i for i, c in enumerate(self.mnist_selected_classes)}
        
        # Filter CIFAR-10 to selected classes
        self._filter_cifar_classes()
        
        # Build the domino dataset
        self.data, self.targets, self.spurious_targets = self._build_spurious_dataset()
        
    def _organize_mnist_by_digit(self):
        mnist_targets = np.array(self.mnist_dataset.targets)
        
        indices = np.arange(len(mnist_targets))
        
        # Group indices by digit
        self.mnist_class_idxs = defaultdict(list)
        for digit in self.mnist_selected_classes:
            mask = mnist_targets == digit
            self.mnist_class_idxs[self.mnist_class2idx[digit]] = indices[mask]
            
                
    def _filter_cifar_classes(self):
        """Filter CIFAR-10 to only include selected classes."""
        if len(self.selected_classes) < 10:
            mask = np.isin(self.cifar_dataset.targets, self.selected_classes)
            self.cifar_data = self.cifar_dataset.data[mask]
            self.cifar_targets = np.array(self.cifar_dataset.targets)[mask]
        else:
            self.cifar_data = self.cifar_dataset.data
            self.cifar_targets = np.array(self.cifar_dataset.targets)
    
    def _build_spurious_dataset(self):
        """Build dataset with exact spurious correlations."""
        n_samples = len(self.cifar_data)
        spurious_targets = np.zeros(n_samples, dtype=int)
        
        # Remap CIFAR targets to new indices
        remapped_targets = np.array([self.class2idx[t] for t in self.cifar_targets])
        
        # Generate spurious labels (MNIST digits)
        spurious_targets = self._generate_spurious_labels(remapped_targets)
        
        # Organize MNIST data by digit for efficient sampling
        self._organize_mnist_by_digit()
        
        # Create domino images
        domino_images = self._create_domino_batch(
            self.cifar_data, spurious_targets
        )
        
        return domino_images, remapped_targets, spurious_targets
    
    def _generate_spurious_labels(self, class_indices):
        """Generate spurious labels using exact correlation percentages."""
        n_samples = len(class_indices)
        spurious_targets = np.zeros(n_samples, dtype=int)
        
        # Process each class
        for class_idx in range(self.n_classes):
            class_mask = (class_indices == class_idx)
            n_class_samples = np.sum(class_mask)
            
            if n_class_samples == 0:
                continue
                
            # Calculate exact counts
            n_majority = int(n_class_samples * self.spurious_correlation)
            n_minority = n_class_samples - n_majority
            
            # Create spurious labels for this class
            class_spurious = np.zeros(n_class_samples, dtype=int)
            
            # Majority: MNIST digit matches CIFAR class
            class_spurious[:n_majority] = class_idx
            
            # Minority: balanced distribution of other digits
            if n_minority > 0:
                other_digits = np.array([i for i in range(self.n_classes) if i != class_idx])
                
                if len(other_digits) > 0:
                    minority_distribution = np.repeat(
                        other_digits, 
                        n_minority // len(other_digits)
                    )
                    
                    remainder = n_minority % len(other_digits)
                    if remainder > 0:
                        extra = np.array(other_digits[:remainder])
                        minority_distribution = np.concatenate([minority_distribution, extra])
                    
                    class_spurious[n_majority:] = minority_distribution
            
            # Shuffle within class
            np.random.shuffle(class_spurious)
            
            # Assign back to main array
            spurious_targets[class_mask] = class_spurious
        
        return spurious_targets
    
    def _sample_mnist_images(self, spurious_targets):
        """Sample MNIST images based on spurious targets (digits)."""
        n_samples = len(spurious_targets)
        mnist_indices = np.zeros(n_samples, dtype=int)
        
        # Get unique digits and their positions
        unique_digits, inverse_indices = np.unique(spurious_targets, return_inverse=True)
        
        # Process each unique digit in batches
        for digit in unique_digits:
            # Find all positions that need this digit
            digit_mask = (spurious_targets == digit)
            n_needed = np.sum(digit_mask)
            
            if len(self.mnist_class_idxs[digit]) > 0:
                # Vectorized random sampling for all instances of this digit
                sampled_indices = np.random.choice(
                    self.mnist_class_idxs[digit], 
                    size=n_needed, 
                    replace=True
                )
            else:
                raise ValueError(f"Did not found any digit {digit}.")
            
            # Assign sampled indices to correct positions
            mnist_indices[digit_mask] = sampled_indices
        
        return mnist_indices

    
    def _create_domino_batch(self, cifar_images, spurious_targets):
        """Create batch of domino images by stacking CIFAR-10 and MNIST - vectorized version."""
        
        n_samples = len(cifar_images)
        
        # Sample corresponding MNIST images
        mnist_indices = self._sample_mnist_images(spurious_targets)
        
        # Domino: 32 (CIFAR) + 32 (padded MNIST) = 64 height, 32 width, 3 channels
        domino_images = np.zeros((n_samples, 64, 32, 3), dtype=np.uint8)
        
        # Get all MNIST data at once
        mnist_raw = self.mnist_dataset.data.numpy()[mnist_indices]  # (n_samples, 28, 28)
        
        mnist_images = self._prepare_mnist_batch(mnist_raw)
        
        domino_images[:, :32, :, :] = cifar_images      # Top half: CIFAR
        domino_images[:, 32:, :, :] = mnist_images    # Bottom half: MNIST
        
        return domino_images

    def _prepare_mnist_batch(self, mnist_batch):
        """Convert batch of MNIST images to RGB and pad to 32x32 - vectorized version."""
        # Convert grayscale to RGB by repeating across channel dimension
        # Shape: (n_samples, 28, 28) -> (n_samples, 28, 28, 3)
        mnist_rgb = np.stack([mnist_batch] * 3, axis=-1)
        
        # Vectorized padding: add 2 pixels on each side (28x28 -> 32x32)
        # Padding format: ((batch, batch), (top, bottom), (left, right), (channels, channels))
        mnist_padded = np.pad(
            mnist_rgb, 
            ((0, 0), (2, 2), (2, 2), (0, 0)), 
            mode='constant', 
            constant_values=0
        )  # Shape: (n_samples, 32, 32, 3)
        
        return mnist_padded.astype(np.uint8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Get a single domino sample."""
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Get spurious label and group
        spurious_label = int(self.spurious_targets[index])
        
        # Group: positive if majority (aligned), negative if minority (misaligned)
        group = target + 1 if target == spurious_label else -target - 1
        
        return img, target, group, spurious_label
    
    def get_group_stats(self, verbose=True):
        """Print statistics about the groups in the dataset."""
        if verbose:
            print(f"Total samples: {len(self)}")
            print("Group statistics (CIFAR class vs MNIST digit):")
        
        stats = {}
        for i in range(len(self)):
            target = self.targets[i]
            spurious = self.spurious_targets[i]
            cifar_class = self.selected_classes[target]
            
            key = (cifar_class, spurious)
            stats[key] = stats.get(key, 0) + 1
        
        if verbose:
            for (cifar_class, mnist_digit), count in sorted(stats.items()):
                alignment = "majority" if cifar_class == mnist_digit else "minority"
                class_name = self.CLASS_NAMES[cifar_class]
                print(f"  CIFAR {cifar_class} ({class_name}) + MNIST {mnist_digit}: {count} samples ({alignment})")
        
        return stats


def get_spurious_domino_datasets(
    root,
    spurious_correlation,
    transform=None,
    val_ratio=0.0,
    selected_classes=None,
    seed=42,
    setup='known',
    **kwargs
):
    """Create train/val/test datasets with spurious correlations."""
    
    if transform is None:
        # Note: domino images are 64x32, adjust normalization if needed
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    n_classes = len(selected_classes) if selected_classes is not None else 10
    balanced_correlation = 1.0 / n_classes
    
    test_dataset = SpuriousCorrelationDomino(
        root,
        train=False,
        transform=transform,
        download=True,
        spurious_correlation=balanced_correlation,
        selected_classes=selected_classes,
        seed=seed,
        **kwargs
    )
    
    train_dataset = SpuriousCorrelationDomino(
        root,
        train=True,
        transform=transform,
        download=True,
        spurious_correlation=spurious_correlation,
        selected_classes=selected_classes,
        seed=seed,
        **kwargs
    )
    
    if val_ratio > 0:
        if setup == 'known':
            val_dataset = SpuriousCorrelationDomino(
                root,
                train=True,
                transform=transform,
                spurious_correlation=balanced_correlation,
                selected_classes=selected_classes,
                seed=seed,
                **kwargs
            )
        else:
            val_dataset = train_dataset
            
        indices = list(range(len(train_dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(val_ratio * len(indices))
        val_dataset = torch.utils.data.Subset(val_dataset, indices[:split])
        train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
        return train_dataset, val_dataset, test_dataset
    
    return train_dataset, test_dataset


def get_spurious_domino_dataloaders(
    root,
    batch_size,
    spurious_correlation,
    val_ratio=0.0,
    selected_classes=None,
    **kwargs
):
    """Create DataLoaders for spurious Domino dataset."""
    
    dataloader_kwargs = {"num_workers": 2, "persistent_workers": True}
    # Extract dataset-specific kwargs
    dataset_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['seed', 'setup', 'transform', 'mnist_selected_classes']}
    
    # Update dataloader kwargs with remaining kwargs
    dataloader_kwargs.update({k: v for k, v in kwargs.items() 
                             if k not in dataset_kwargs})
    
    if val_ratio > 0.0:
        train_dataset, val_dataset, test_dataset = get_spurious_domino_datasets(
            root, spurious_correlation, val_ratio=val_ratio, selected_classes=selected_classes, **dataset_kwargs
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs),
        )
    else:
        train_dataset, test_dataset = get_spurious_domino_datasets(
            root, spurious_correlation, val_ratio=val_ratio, selected_classes=selected_classes, **dataset_kwargs
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs),
        )
