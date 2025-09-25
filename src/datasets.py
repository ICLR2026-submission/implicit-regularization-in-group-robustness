import src.cifar10_dataset as cifar10_dataset
import src.cmnist_dataset as cmnist_dataset
import src.domino_dataset as domino_dataset
import src.waterbirds_dataset as waterbirds_dataset
import src.celeba_dataset as celeba_dataset

import numpy as np
import random

import torch
from torch.utils.data import DataLoader



def get_dataset_class(name):
    match name:
        case "cifar10":
            return cifar10_dataset.SpuriousCorrelationCIFAR10
        case "domino":
            return domino_dataset.SpuriousCorrelationDomino
        case "cmnist":
            return cmnist_dataset.ColourBiasedMNIST
        case "celeba":
            return celeba_dataset.CelebADataset
        case "waterbirds":
            return waterbirds_dataset.WaterbirdsDataset
        case _:
            return "Invalid day number"


class DatasetGetter:
    def __init__(
            self,
            dataset_name,
            root,
            spurious_correlation,
            val_ratio=0.0,
            selected_classes=None,
            seed=42,):
        self.dataset_name = dataset_name
        self.root = root
        self.spurious_correlation = spurious_correlation
        self.dataset_class = get_dataset_class(dataset_name)
        self.n_classes = len(selected_classes) if selected_classes is not None else 10
        self.balanced_correlation = 1.0 / self.n_classes
        self.val_ratio = val_ratio
        self.seed = seed
        self.common_args = {'root': self.root, 'download':True, 'selected_classes':selected_classes, 'seed':self.seed,}
        self.train_indices = None


    def get_dataset(
        self,
        split,
        setup=None,
        transform=None,
        **kwargs,
        ):
        assert not (split == 'train' and setup == 'known')
        sp = self.balanced_correlation if setup == 'known' else self.spurious_correlation

        if self.dataset_name in ['waterbirds', 'celeba']:
            return self.dataset_class(kwargs['args'], split, setup)

        elif split == 'train' or split == 'val':
            train_dataset = self.dataset_class(
                train=True,
                transform=transform,
                spurious_correlation=sp,
                **self.common_args,
                **kwargs,
            )
            if self.train_indices is None:
                # splitting train and val
                indices = list(range(len(train_dataset)))
                np.random.seed(self.seed)
                np.random.shuffle(indices)
                val_split_idx = int(self.val_ratio * len(indices))
                self.val_indices = indices[:val_split_idx]
                self.train_indices = indices[val_split_idx:]
            
            if split == 'train':
                return torch.utils.data.Subset(train_dataset, self.train_indices)
            else:
                return torch.utils.data.Subset(train_dataset, self.val_indices)

        elif split == 'test':
            return self.dataset_class(
                train=False,
                transform=transform,
                spurious_correlation=sp,
                **self.common_args,
                **kwargs,
            )
        else:
            raise ValueError("Wrong Split")