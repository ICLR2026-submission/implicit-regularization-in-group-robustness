"""
Color MNIST Dataset. Adapted from https://github.com/clovaai/rebias
"""

import os
import numpy as np
from PIL import Image
import random

import torch
from torch.utils import data
from torch.utils.data import DataLoader


from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler


class ColourBiasedMNIST(MNIST):
    """
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    spurious_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        spurious_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP1 = [
        [230, 25, 75],  # Vivid Red
        [60, 180, 75],  # Vivid Green
        [255, 225, 25],  # Vivid Yellow
        [0, 130, 200],  # Vivid Blue
        [245, 130, 48],  # Vivid Orange
        [145, 30, 180],  # Vivid Purple
        [70, 240, 240],  # Cyan
        [240, 50, 230],  # Magenta
        [210, 245, 60],  # Lime
        [250, 190, 190],  # Light Pink
    ]

    def __init__(
        self,
        root,
        cmap=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        spurious_correlation=1.0,
        n_confusing_labels=None,
        colored_numbers=True,
        selected_classes=None,
        seed=0,
        **kwargs,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        random.seed(seed)
        np.random.seed(seed)

        self.random = True
        self.spurious_correlation = spurious_correlation
        self.colored_numbers = colored_numbers

        self.selected_classes = (
            np.arange(10) if selected_classes is None else selected_classes
        )
        self.n_classes = len(self.selected_classes)
        self.n_labels_complete = self.targets.max().item() + 1
        assert 0 < self.n_classes <= self.n_labels_complete
        assert np.all(np.isin(self.selected_classes, np.arange(self.n_labels_complete)))
        self.class2idx = {c: i for i, c in enumerate(self.selected_classes)}

        self.cmap = cmap if cmap is not None else self.COLOUR_MAP1
        self.cmap = np.array(self.cmap)[self.selected_classes]

        self.n_confusing_labels = (
            n_confusing_labels if n_confusing_labels is not None else self.n_classes - 1
        )
        assert 1 <= self.n_confusing_labels < self.n_classes

        self.data, self.targets, self.spurious_targets = self.build_biased_mnist()
        indices = np.arange(len(self.data))
        # self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.spurious_targets = self.spurious_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _update_bias_indices(self, bias_indices, label, idx):
        indices = np.where((self.targets == label).numpy())[0]
        # self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.spurious_correlation)
        n_decorrelated_per_class = int(
            np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels))
        )
        correlated_indices = indices[:n_correlated_samples]
        bias_indices[idx] = torch.cat([bias_indices[idx], correlated_indices])

        decorrelated_indices = torch.split(
            indices[n_correlated_samples:], n_decorrelated_per_class
        )
        other_labels = [
            _label % self.n_classes
            for _label in range(idx + 1, idx + 1 + self.n_confusing_labels)
        ]
        # self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST."""

        # bias_indices = {label:  for label in self.selected_classes}
        bias_indices = {
            self.class2idx[c]: torch.LongTensor() for c in self.selected_classes
        }
        for c in self.selected_classes:
            idx = self.class2idx[c]
            self._update_bias_indices(bias_indices, c, idx)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        spurious_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label, self.cmap)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            spurious_targets.extend([bias_label] * len(indices))

        spurious_targets = torch.LongTensor(spurious_targets)
        return data, targets, spurious_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        target = self.class2idx[target]
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sp_label = int(self.spurious_targets[index])
        group = target + 1 if target == sp_label else -target - 1
        return img, target, group, sp_label

    def _binary_to_colour(self, data, color):
        color = torch.ByteTensor(color).view(3, 1, 1)
        # fig = torch.zeros_like(data)
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 1
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        if self.colored_numbers:
            # color foreground
            fg_data *= color
            data = fg_data
        else:
            # color background
            fg_data *= 255
            bg_data = torch.zeros_like(data)
            bg_data[data == 0] = 1
            bg_data[data != 0] = 0
            bg_data = torch.stack([bg_data, bg_data, bg_data], dim=1)
            bg_data = bg_data * color
            data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label, cmap):
        color = cmap[label]
        return self._binary_to_colour(self.data[indices], color), self.targets[indices]


def get_colored_mnist_datasets(
    root,
    spurious_correlation,
    transform=None,
    val_ratio=0.0,
    selected_classes=None,
    seed=0,
    setup='known'
):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    balanced_corr = 1 / (len(selected_classes) if selected_classes is not None else 10)
    test_dataset = ColourBiasedMNIST(
        root,
        train=False,
        transform=transform,
        download=True,
        spurious_correlation=balanced_corr,
        selected_classes=selected_classes,
    )
    train_dataset = ColourBiasedMNIST(
        root,
        train=True,
        transform=transform,
        download=True,
        spurious_correlation=spurious_correlation,
        selected_classes=selected_classes,
    )
    if val_ratio > 0:
        if setup == 'known':
            val_dataset = ColourBiasedMNIST(
                root,
                train=True,
                transform=transform,
                download=True,
                spurious_correlation=balanced_corr,
                selected_classes=selected_classes,
            )
        else: #uknown
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


def get_colored_mnist_dataloaders(
    root,
    batch_size,
    spurious_correlation,
    val_ratio=0.0,
    selected_classes=None,
    **kwargs
):
    dataloader_kwargs = {"num_workers": 2, "persistent_workers": True}
    dataloader_kwargs.update(kwargs)

    if val_ratio > 0.0:
        train_dataset, val_dataset, test_dataset = get_colored_mnist_datasets(
            root, spurious_correlation, None, val_ratio, selected_classes, **kwargs,
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
        train_dataset, test_dataset = get_colored_mnist_datasets(
            root, spurious_correlation, None, val_ratio, selected_classes
        )
        return (
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
            ),
            DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
            ),
        )
