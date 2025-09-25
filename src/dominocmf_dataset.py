import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from itertools import product


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [1, h, w])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((2, h, w), dtype=dtype)], axis=0)
  else:
    arr = np.concatenate([np.zeros((1, h, w), dtype=dtype),
                          arr,
                          np.zeros((1, h, w), dtype=dtype)], axis=0)
  return arr


def keep_only_lbls(dataset, lbls):
  lbls = {lbl: i for i, lbl in enumerate(lbls)}
  final_X, final_Y = [], []
  for x, y in dataset:
    if y in lbls:
      final_X.append(x)
      final_Y.append(lbls[y])
  X = torch.stack(final_X)
  Y = torch.tensor(final_Y).float().view(-1,1)
  return X, Y


def apply_color_to_imgs(imgs, is_red=True):
    colored_data = []
    for idx, img in enumerate(imgs):
        img = color_grayscale_arr(img.squeeze().numpy(), red=is_red)
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)), constant_values=0)
        colored_data.append(torch.tensor(img))
    colored_data = torch.stack(colored_data)
    return colored_data


# at least 5000
class DominoCMFDataset(Dataset):
    """
    This is the dataset class that creates a datset with spurious attributes as follows:
    The shape attribute and color attribute each seperately correlate to the lables. The shape attribute is the visible spurious attribute,
     however, the color attribute is hidden.
    
    - Please make sure the input datasets are seperated before being passed to this class to ensure data leak does not happen.
    - It is recommended for each class to have at least 5000 datapoints.

    Parameters
    ----------
    split (Optional):
        name of the split. only used for printing.
    total_count:
        Total count of datapoints in the dataset.
    mnist_X: 
        Seperated parts of torchvision.datasets.MNIST dataset.
    fmnist_X: 
        Seperated parts of torchvision.datasets.FashionMNIST dataset.
    fmnist_X: 
        Seperated parts of torchvision.datasets.CIFAR10 dataset.
    shape_correlation:
        Correlation between shape(MNIST and FashionMNIST) and CIFAR10.
    color_correlation:
        Correlation between color(red and green) and CIFAR10.
    group_policy:
        This variable decides how groups are labeled. There are three posibilities:
            1. shape: Only use shape label for constructing group labels.
            1. color: Only use color label for constructing group labels.
            1. combined: Use both shape and color labels for constructing group labels.
    """

    CIFAR_LABELS = [
        'CARS',
        'ANIMALS'
    ]

    SHAPE_LABELS = [
        'MNIST',
        'FMNIST'
    ]

    COLOR_LABELS = [
        'RED',
        'GREEN'
    ]

    def __init__(
        self,
        total_count,
        X_mnist,
        X_fmnist,
        X_c_cars,
        X_c_animals,
        shape_correlation=0.5,
        color_correlation=0.5,
        group_policy='shape',
        split=None
        ):

        if split is not None:
            print(f'{split} dataset:')

        assert  (total_count <= len(X_mnist)) and \
                (total_count <= len(X_fmnist)) and \
                (np.ceil(total_count / 2) <= len(X_c_cars)) and \
                (np.ceil(total_count / 2) <= len(X_c_animals))

        # calculate each group counts
        group_count = self._calculate_each_group_data_counts(total_count, shape_correlation, color_correlation)

        print(f'\tTotal count:', total_count)
        print(*[f'\t\tgroup {key}: {value}' for key, value in group_count.items()], sep='\n')

        # pick cifar images for each group
        images = self._create_images(X_mnist, X_fmnist, X_c_animals, X_c_cars, group_count)

        # create group dict for determining group label of each datapoint.
        group_dict = self._create_group_dict(group_policy)

        # create spurious label policy dict for determining each spurious label of each datapoint.
        sp_label_dict = self._create_spurious_label_dict(group_policy)

        print(f'\tAssigned group numbers:')
        print(*[f'\t\tgroup {key}: {value}' for key, value in group_dict.items()], sep='\n')

        print(f'\tAssigned spurious numbers:')
        print(*[f'\t\tgroup {key}: {value}' for key, value in sp_label_dict.items()], sep='\n')

        # create the dataset
        self.images = torch.cat([
            images[group]
            for group in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS)
        ])

        self.labels = torch.cat([
            torch.Tensor([self.CIFAR_LABELS.index(group[0])] * group_count[group])
            for group in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS)
        ]).long()

        self.groups = torch.cat([
            torch.tensor([group_dict[group]] * group_count[group])
            for group in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS)
        ]).long()

        self.sp_labels = torch.cat([
            torch.tensor([sp_label_dict[group]] * group_count[group])
            for group in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS)
        ]).long()

        self.n_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        y = self.labels[idx]
        group = self.groups[idx]
        sp_label = self.sp_labels[idx]
        return image, y, group, sp_label

    def _calculate_each_group_data_counts(self, total_count, shape_correlation, color_correlation):
        group_count = {}
        for idx, (cifar, shape, color) in enumerate(product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS)):
            # make sure true labels are balanced
            count = 0.5 * total_count
            # apply shape correlation
            count = count * (shape_correlation if self.SHAPE_LABELS.index(shape) == self.CIFAR_LABELS.index(cifar) else 1 - shape_correlation)
            # apply color correlation
            count = count * (color_correlation if self.COLOR_LABELS.index(color) == self.CIFAR_LABELS.index(cifar) else 1 - color_correlation)
            # convert float to int
            count = int(np.ceil(count)) if self.COLOR_LABELS.index(color) == self.SHAPE_LABELS.index(shape) else int(count)
            group_count[(cifar, shape, color)] = count
        return group_count

    def _create_images(self, X_mnist, X_fmnist, X_c_animals, X_c_cars, group_count):
        # apply permutation to both datasets so that different classes are dispersed evenly
        X_mnist = X_mnist[torch.randperm(X_mnist.shape[0], generator=torch.Generator().manual_seed(21))]
        X_fmnist = X_fmnist[torch.randperm(X_fmnist.shape[0], generator=torch.Generator().manual_seed(21))]
        images = {}
        for cifar in self.CIFAR_LABELS:
            X_cifar = X_c_cars if cifar == 'CARS' else X_c_animals
            current_idx = 0
            for shape in self.SHAPE_LABELS:
                source = X_mnist if shape == 'MNIST' else X_fmnist
                for color in self.COLOR_LABELS:
                    count = group_count[(cifar, shape, color)]
                    picked_imgs, source = source[: count], source[count:]
                    colored_imgs = apply_color_to_imgs(picked_imgs, is_red=color=='RED')
                    images[(cifar, shape, color)] = torch.cat((X_cifar[current_idx: current_idx + count], colored_imgs), dim=2)
                    current_idx += count
                if shape == 'MNIST':
                    X_mnist = source
                else:
                    X_fmnist = source
        return images

    def _create_group_dict(self, group_policy):
        assert group_policy in ['color', 'shape', 'combined']
        group_dict = {}
        for (cifar, shape, color) in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS):
            group_label = self.CIFAR_LABELS.index(cifar)
            if group_policy != 'color':
                group_label = group_label * 2 + self.SHAPE_LABELS.index(shape)
            if group_policy != 'shape':
                group_label = group_label * 2 + self.COLOR_LABELS.index(color)
            group_dict[(cifar, shape, color)] = group_label
        return group_dict

    def _create_spurious_label_dict(self, group_policy):
        assert group_policy in ['color', 'shape', 'combined']
        sp_label_dict = {}
        for (cifar, shape, color) in product(self.CIFAR_LABELS, self.SHAPE_LABELS, self.COLOR_LABELS):
            sp_label = 0
            if group_policy != 'color':
                sp_label = sp_label * 2 + self.SHAPE_LABELS.index(shape)
            if group_policy != 'shape':
                sp_label = sp_label * 2 + self.COLOR_LABELS.index(color)
            sp_label_dict[(cifar, shape, color)] = sp_label
        return sp_label_dict


def split_dataset_in_two(proportions, dataset, seed=42):
    first, second = random_split(dataset, [proportions, 1-proportions], generator=torch.Generator().manual_seed(seed))
    return first[:], second[:]

def get_domino_cmf_datasets(args):

    mnist_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=0.485,
                                           std=0.229),
        ])
    cifar_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
        ])
    
    MNIST_LABELS_TO_KEEP = [0, 1, 2, 3]
    FMNIST_LABELS_TO_KEEP = [0, 3, 4, 6]
    CIFAR_CARS_LABELS = [0, 1, 8, 9]
    CIFAR_ANIMALS_LABELS = [3, 4, 5, 7]

    mnist_train = MNIST(args.mnist_datapath, train=True, download=True, transform=mnist_transform)
    fmnist_train = FashionMNIST(args.fmnist_datapath, train=True, download=True, transform=mnist_transform)
    cifar10_train = torchvision.datasets.CIFAR10(args.cifar_datapath, train=True, download=True, transform=cifar_transform)

    mnist_train, _ = keep_only_lbls(mnist_train, lbls=MNIST_LABELS_TO_KEEP)
    fmnist_train, _ = keep_only_lbls(fmnist_train, lbls=FMNIST_LABELS_TO_KEEP)
    cifar10_train_cars, _ = keep_only_lbls(cifar10_train, lbls=CIFAR_CARS_LABELS)
    cifar10_train_animals, _ = keep_only_lbls(cifar10_train, lbls=CIFAR_ANIMALS_LABELS)

    mnist_train, mnist_val = split_dataset_in_two(0.5, mnist_train, seed=args.random_seed)
    fmnist_train, fmnist_val = split_dataset_in_two(0.5, fmnist_train, seed=args.random_seed)
    cifar10_train_cars, cifar10_val_cars = split_dataset_in_two(0.5, cifar10_train_cars, seed=args.random_seed)
    cifar10_train_animals, cifar10_val_animals = split_dataset_in_two(0.5, cifar10_train_animals, seed=args.random_seed)

    mnist_test = MNIST(args.mnist_datapath, train=False, download=True, transform=mnist_transform)
    fmnist_test = FashionMNIST(args.fmnist_datapath, train=False, download=True, transform=mnist_transform)
    cifar10_test = torchvision.datasets.CIFAR10(args.cifar_datapath, train=False, download=True, transform=cifar_transform)

    mnist_test, _ = keep_only_lbls(mnist_test, lbls=MNIST_LABELS_TO_KEEP)
    fmnist_test, _ = keep_only_lbls(fmnist_test, lbls=FMNIST_LABELS_TO_KEEP)
    cifar10_test_cars, _ = keep_only_lbls(cifar10_test, lbls=CIFAR_CARS_LABELS)
    cifar10_test_animals, _ = keep_only_lbls(cifar10_test, lbls=CIFAR_ANIMALS_LABELS)

    train_dataset = DominoCMFDataset(
        total_count=args.domniocmf_train_count,
        X_mnist=mnist_train,
        X_fmnist=fmnist_train,
        X_c_cars=cifar10_train_cars,
        X_c_animals=cifar10_train_animals,
        shape_correlation=args.dominocmf_shape_correlation,
        color_correlation=args.dominocmf_color_correlation,
        split='train'
    )

    val_dataset = DominoCMFDataset(
        total_count=args.domniocmf_val_count,
        X_mnist=mnist_val,
        X_fmnist=fmnist_val,
        X_c_cars=cifar10_val_cars,
        X_c_animals=cifar10_val_animals,
        shape_correlation=args.dominocmf_shape_correlation if args.setup == 'unknown' else 0.5,
        color_correlation=args.dominocmf_color_correlation if args.setup == 'unknown' else 0.5,
        group_policy=args.dominocmf_val_group_policy,
        split='validation'
    )

    test_dataset = DominoCMFDataset(
        total_count=args.domniocmf_test_count,
        X_mnist=mnist_test,
        X_fmnist=fmnist_test,
        X_c_cars=cifar10_test_cars,
        X_c_animals=cifar10_test_animals,
        group_policy=args.dominocmf_test_group_policy,
        split='test'
    )

    return train_dataset, val_dataset, test_dataset
    