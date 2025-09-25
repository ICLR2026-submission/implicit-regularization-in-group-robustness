import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from wilds import get_dataset

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

def get_transforms(split, normalize_stats=None,
                   target_resolution=(224, 224),
                   resize_resolution=(256, 256)):

    transform_list = []

    if split == "train":
        transform_list.extend([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip()
        ])
    else:
        transform_list.extend([
            transforms.Resize(resize_resolution),
            transforms.CenterCrop(target_resolution)
        ])

    # Add Tensor conversion and normalization
    transform_list.append(transforms.ToTensor())
    if normalize_stats is not None:
        transform_list.append(transforms.Normalize(*normalize_stats))

    return transforms.Compose(transform_list)

class WaterbirdsDataset(Dataset):
    def __init__(self, args, split, setup='known', **kwargs):
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.env_dict = {(0, 0): 2, (0, 1): -1, (1, 0): -2, (1, 1): 1}  # (y, place)
        self.split = split
        self.dataset_name = "waterbirds"
        self.dataset_dir = os.path.join(args.waterbirds_datapath)
        self.random_seed = args.random_seed
        # Load metadata
        metadata_path = os.path.join(self.dataset_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            _ = get_dataset(dataset=self.dataset_name, download=True)
            self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name + "_v1.0")

        self.metadata_df = pd.read_csv(metadata_path)
        self.metadata_df = self.metadata_df[self.metadata_df["split"] == self.split_dict[split]]

        # Extract labels
        self.y_array = self.metadata_df["y"].values
        self.place_array = self.metadata_df["place"].values
        self.filename_array = self.metadata_df["img_filename"].values
        self.transform = get_transforms(self.split, normalize_stats=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


        if split == "train":
            self.subsample(args.spurious_correlation)
            
        if (split == "val") and (setup == 'unknown'):
            self.subsample(args.spurious_correlation)

        if (split == "test") and (setup == 'unknown'):
            self.subsample(args.spurious_correlation)

    def subsample(self, spurious_correlation):
        spurious_correlation = int(spurious_correlation * 100)
        
        np.random.seed(self.random_seed)
        min_idxs = []
        maj_idxs = []

        for idx, (y, place) in enumerate(zip(self.y_array, self.place_array)):
            if (y, place) in [(1, 0), (0, 1)]:  # minority groups
                min_idxs.append(idx)
            else:
                maj_idxs.append(idx)

        min_idxs = np.array(min_idxs)
        maj_idxs = np.array(maj_idxs)
        min_count = len(min_idxs)
        maj_count = len(maj_idxs)

        current_corr = 100 * len(maj_idxs) / (len(maj_idxs) + len(min_idxs))
        print(f"[Before subsampling] spurious_correlation: {current_corr:.2f}%, Target: {spurious_correlation}%")

        if spurious_correlation is None or abs(current_corr - spurious_correlation) < 0.5:
            return

        for i in range(100):
            if current_corr > spurious_correlation:
                b = min_count - i
                if b <= 0:
                    break
                a = (spurious_correlation * b) // (100 - spurious_correlation)
            else:
                a = maj_count - i
                if a <= 0:
                    break
                b = ((100 - spurious_correlation) * a) // spurious_correlation
            if b > len(min_idxs) or a > len(maj_idxs):
                continue
            chosen_min = min_idxs[:b]
            chosen_maj = maj_idxs[:a]
            break

        selected = np.concatenate([chosen_min, chosen_maj])
        self.y_array = self.y_array[selected]
        self.place_array = self.place_array[selected]
        self.filename_array = self.filename_array[selected]

        print(f"[After subsampling] Maj: {len(chosen_maj)}, Min: {len(chosen_min)}, spurious_correlation: {100 * len(chosen_maj)/(len(selected)):.2f}%")

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_path = os.path.join(self.dataset_dir, self.filename_array[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, y, self.env_dict[(y, place)], place
