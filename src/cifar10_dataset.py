"""
CIFAR-10 Dataset with Spurious Correlation via Colored Corner Boxes
"""

import os
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from collections import Counter


class SpuriousCorrelationCIFAR10(CIFAR10):
    """
    CIFAR-10 dataset with spurious correlation created by adding colored boxes to image corners.

    Each class gets assigned a specific color, and the correlation parameter controls how often
    samples from that class get their designated color vs. a random different color.

    Parameters
    ----------
    root : str
        Path to CIFAR-10 dataset.
    spurious_correlation : float, default=0.95
        Controls the level of spurious correlation between class and corner box color.
        - 1.0: Perfect correlation (each class always gets its designated color)
        - 0.1: Minimal correlation (roughly uniform color distribution)
    box_size : int, default=3
        Size of the colored box added to the corner (box_size x box_size pixels)
    selected_classes : list, optional
        Subset of CIFAR-10 classes to use (0-9). If None, uses all classes.
    box_position : str, default='top_left'
        Position of the colored box: 'random', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """

    CLASS_NAMES = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Default color map
    DEFAULT_COLOR_MAP = [
        [230, 25, 75],  # Red - plane
        [60, 180, 75],  # Green - car
        [255, 225, 25],  # Yellow - bird
        [0, 130, 200],  # Blue - cat
        [245, 130, 48],  # Orange - deer
        [145, 30, 180],  # Purple - dog
        [70, 240, 240],  # Cyan - frog
        [240, 50, 230],  # Magenta - horse
        [210, 245, 60],  # Lime - ship
        [250, 190, 190],  # Pink - truck
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        spurious_correlation=0.95,
        box_size=3,
        selected_classes=None,
        box_position="top_left",
        color_map=None,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        self.spurious_correlation = spurious_correlation
        self.box_size = box_size
        self.box_position = box_position

        # Handle class selection
        self.selected_classes = (
            list(range(10)) if selected_classes is None else selected_classes
        )
        self.n_classes = len(self.selected_classes)
        assert 0 < self.n_classes <= 10
        assert all(c in range(10) for c in self.selected_classes)

        # Create class mapping
        self.class2idx = {c: i for i, c in enumerate(self.selected_classes)}

        # Set up color mapping
        self.color_map = color_map if color_map is not None else self.DEFAULT_COLOR_MAP
        self.color_map = (
            np.array(self.color_map)[self.selected_classes]
            if len(self.color_map) > self.n_classes
            else self.color_map
        )

        # Filter dataset to selected classes and build spurious correlation
        self._filter_classes()
        self.data, self.targets, self.spurious_targets = self._build_spurious_dataset()

    def _filter_classes(self):
        """Filter dataset to only include selected classes."""
        if len(self.selected_classes) < 10:
            # Create mask for selected classes
            mask = np.isin(self.targets, self.selected_classes)
            self.data = self.data[mask]
            self.targets = np.array(self.targets)[mask]

    def _build_spurious_dataset(self):
        """Build dataset with exact spurious correlations."""
        n_samples = len(self.data)
        spurious_targets = np.zeros(n_samples, dtype=int)
        modified_data = self.data.copy()

        # Remap targets to new indices
        remapped_targets = np.array([self.class2idx[t] for t in self.targets])

        # Create spurious labels vector efficiently
        spurious_targets = self._generate_spurious_labels_vectorized(remapped_targets)

        # Apply colored boxes vectorized by spurious label
        for spurious_idx in range(self.n_classes):
            mask = spurious_targets == spurious_idx
            if np.any(mask):
                color = self.color_map[spurious_idx]
                modified_data[mask] = self._add_colored_box_vectorized(
                    modified_data[mask], color
                )

        return modified_data, remapped_targets, spurious_targets

    def _generate_spurious_labels_vectorized(self, class_indices):
        """Generate spurious labels."""
        n_samples = len(self.targets)
        spurious_targets = np.zeros(n_samples, dtype=int)

        # Process each class
        for class_idx in range(self.n_classes):
            class_mask = class_indices == class_idx
            n_class_samples = np.sum(class_mask)

            if n_class_samples == 0:
                continue

            # Calculate counts
            n_majority = int(n_class_samples * self.spurious_correlation)
            n_minority = n_class_samples - n_majority

            # Create spurious labels for this class
            class_spurious = np.zeros(n_class_samples, dtype=int)

            # Majority samples get their own class color
            class_spurious[:n_majority] = class_idx

            # Minority samples get balanced other colors
            if n_minority > 0:
                other_classes = np.array(
                    [i for i in range(self.n_classes) if i != class_idx]
                )

                # Distribute minority samples evenly across other classes
                minority_distribution = np.repeat(
                    other_classes, n_minority // len(other_classes)
                )

                # Handle remainder
                remainder = n_minority % len(other_classes)
                if remainder > 0:
                    extra = other_classes[:remainder]
                    minority_distribution = np.concatenate(
                        [minority_distribution, extra]
                    )

                class_spurious[n_majority:] = minority_distribution

            # Shuffle within class to randomize order
            np.random.shuffle(class_spurious)

            # Assign back to main array
            spurious_targets[class_mask] = class_spurious

        return spurious_targets

    def _add_colored_box_vectorized(self, images, color):
        """Apply colored box to multiple images at once."""
        # This processes all images with the same color at once
        modified_images = images.copy()

        # Determine box coordinates once
        h, w = modified_images.shape[1:3]
        n_images = len(modified_images)

        if self.box_position == "random":
            for i in range(n_images):
                # Generate random top-left corner coordinates
                max_y = max(0, h - self.box_size)
                max_x = max(0, w - self.box_size)

                y_start = np.random.randint(0, max_y + 1) if max_y > 0 else 0
                x_start = np.random.randint(0, max_x + 1) if max_x > 0 else 0

                y_end = min(y_start + self.box_size, h)
                x_end = min(x_start + self.box_size, w)

                modified_images[i, y_start:y_end, x_start:x_end] = color
        else:
            if self.box_position == "top_left":
                y_slice = slice(0, min(self.box_size, h))
                x_slice = slice(0, min(self.box_size, w))
            elif self.box_position == "top_right":
                y_slice = slice(0, min(self.box_size, h))
                x_slice = slice(max(0, w - self.box_size), w)
            elif self.box_position == "bottom_left":
                y_slice = slice(max(0, h - self.box_size), h)
                x_slice = slice(0, min(self.box_size, w))
            else:  # bottom_right
                y_slice = slice(max(0, h - self.box_size), h)
                x_slice = slice(max(0, w - self.box_size), w)

            modified_images[:, y_slice, x_slice] = color

        return modified_images

    def __getitem__(self, index):
        """Get a single sample from the dataset."""
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
        groups = self.targets + 1
        groups = np.where(self.targets == self.spurious_targets, groups, -groups)
        group_counts = Counter(groups)

        if verbose:
            print(f"Total samples: {len(self)}")
            print(
                f"Dataset contains {len(group_counts)} unique (target, spurious) groups:"
            )

            for group, count in sorted(group_counts.items()):
                if group > 0:
                    target = group - 1
                    alignment = "majority"
                else:
                    target = -group - 1
                    alignment = "minority"
                class_name = self.CLASS_NAMES[self.selected_classes[target]]
                print(f"  Class {target} ({class_name}, {alignment}): {count} samples")

        return group_counts


def get_spurious_cifar10_datasets(
    root,
    spurious_correlation,
    transform=None,
    val_ratio=0.0,
    selected_classes=None,
    seed=42,
    setup="known",
    **kwargs,
):
    """Create train/val/test datasets with spurious correlations."""

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    # Test set should be unbiased (balanced spurious correlation)
    n_classes = len(selected_classes) if selected_classes is not None else 10
    balanced_correlation = 1.0 / n_classes

    test_dataset = SpuriousCorrelationCIFAR10(
        root,
        train=False,
        transform=transform,
        download=True,
        spurious_correlation=balanced_correlation,
        selected_classes=selected_classes,
        seed=seed,
        **kwargs,
    )
    train_dataset = SpuriousCorrelationCIFAR10(
        root,
        train=True,
        transform=transform,
        download=True,
        spurious_correlation=spurious_correlation,
        selected_classes=selected_classes,
        seed=seed,
        **kwargs,
    )

    if val_ratio > 0:
        if setup == "known":
            val_dataset = SpuriousCorrelationCIFAR10(
                root,
                train=True,
                transform=transform,
                spurious_correlation=balanced_correlation,
                selected_classes=selected_classes,
                seed=seed,
                **kwargs,
            )
        else:  # uknown
            val_dataset = train_dataset
        indices = list(range(len(train_dataset)))
        # shuffle
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(val_ratio * len(indices))
        val_dataset = torch.utils.data.Subset(val_dataset, indices[:split])
        train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
        return train_dataset, val_dataset, test_dataset

    return train_dataset, test_dataset



def get_spurious_cifar10_dataloaders(
    root,
    batch_size,
    spurious_correlation,
    val_ratio=0.0,
    selected_classes=None,
    **kwargs,
):
    """Create DataLoaders for spurious CIFAR-10 dataset."""

    dataloader_kwargs = {"num_workers": 2, "persistent_workers": True}
    # Extract dataset-specific kwargs
    dataset_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["box_size", "box_position", "color_map", "seed", "setup", "transform"]
    }

    # Update dataloader kwargs with remaining kwargs
    dataloader_kwargs.update(
        {k: v for k, v in kwargs.items() if k not in dataset_kwargs}
    )

    if val_ratio > 0.0:
        train_dataset, val_dataset, test_dataset = get_spurious_cifar10_datasets(
            root,
            spurious_correlation,
            val_ratio=val_ratio,
            selected_classes=selected_classes,
            **dataset_kwargs,
        )
        return (
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
            ),
            DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
            ),
            DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
            ),
        )
    else:
        train_dataset, test_dataset = get_spurious_cifar10_datasets(
            root,
            spurious_correlation,
            val_ratio=val_ratio,
            selected_classes=selected_classes,
            **dataset_kwargs,
        )
        return (
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
            ),
            DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
            ),
        )
