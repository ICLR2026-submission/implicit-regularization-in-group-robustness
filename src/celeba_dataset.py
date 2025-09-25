import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from wilds import get_dataset

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_transform_cub(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class CelebADataset(Dataset):
    def __init__(self, args, split, setup='known', **kwargs):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): 0,   # nongrey hair, female
            (0, 1): 1,   # nongrey hair, male
            (1, 0): 2,   # gray hair, female
            (1, 1): 3    # gray hair, male
        }
        self.split = split
        self.dataset_name = 'celebA'
        self.dataset_dir = args.celeba_datapath
        self.random_seed = args.random_seed
        if not os.path.exists(self.dataset_dir):
            _ = get_dataset(dataset=self.dataset_name, download=True)
            self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name + "_v1.0")

        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celeba_metadata.csv'))

        self.y_array = self.metadata_df['y'].values
        self.gender_array = self.metadata_df['place'].values
        self.image_id_array = self.metadata_df['image_id'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = get_transform_cub(self.split=='train')
        # Super-balanced option: force equal subgroup counts in train split
        if getattr(args, 'celeba_super_balanced', 0) == 1 and self.split == 'train':
            desired_per_group = 1387
            rng = np.random.default_rng(self.random_seed)
            indices_per_group = {
                (0, 0): [],
                (0, 1): [],
                (1, 0): [],
                (1, 1): [],
            }
            for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
                indices_per_group[(int(y), int(gender))].append(idx)
            # Sample exactly desired_per_group from each
            selected_indices = []
            for key, idxs in indices_per_group.items():
                if len(idxs) < desired_per_group:
                    raise ValueError(f"CelebA super-balance requested {desired_per_group} for group {key} but only {len(idxs)} available in split '{self.split}'.")
                chosen = rng.choice(idxs, size=desired_per_group, replace=False)
                selected_indices.append(chosen)
            selected_indices = np.concatenate(selected_indices)
            rng.shuffle(selected_indices)
            # Apply selection
            self.y_array = self.y_array[selected_indices]
            self.gender_array = self.gender_array[selected_indices]
            self.filename_array = self.filename_array[selected_indices]
            self.image_id_array = self.image_id_array[selected_indices]
        # Super-balanced option: force equal subgroup counts in validation split
        elif getattr(args, 'celeba_val_super_balanced', 0) == 1 and self.split == 'val':
            desired_per_group = 1387
            rng = np.random.default_rng(self.random_seed)
            indices_per_group = {
                (0, 0): [],
                (0, 1): [],
                (1, 0): [],
                (1, 1): [],
            }
            for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
                indices_per_group[(int(y), int(gender))].append(idx)
            # Sample exactly desired_per_group from each
            selected_indices = []
            for key, idxs in indices_per_group.items():
                if len(idxs) < desired_per_group:
                    raise ValueError(f"CelebA val super-balance requested {desired_per_group} for group {key} but only {len(idxs)} available in split '{self.split}'.")
                chosen = rng.choice(idxs, size=desired_per_group, replace=False)
                selected_indices.append(chosen)
            selected_indices = np.concatenate(selected_indices)
            rng.shuffle(selected_indices)
            # Apply selection
            self.y_array = self.y_array[selected_indices]
            self.gender_array = self.gender_array[selected_indices]
            self.filename_array = self.filename_array[selected_indices]
            self.image_id_array = self.image_id_array[selected_indices]
        elif self.split == 'train':
            self.subsample(args.spurious_correlation)
        if (self.split == 'val') and (setup == 'unknown'):
            self.subsample(args.spurious_correlation)
        if (split == "test") and (setup == 'unknown'):
            self.subsample(args.spurious_correlation)
    
    def subsample(self, spurious_correlation):
        # Define groupings
        spurious_correlation = int(spurious_correlation*100)
        np.random.seed(self.random_seed)
        min_indexes = []
        maj_indexes = []
        for idx, (y, gender) in enumerate(zip(self.y_array, self.gender_array)):
            if (y, gender) in [(1, 0), (0, 1)]:  # minority groups
                min_indexes.append(idx)
            else:  # (0,0), (1,1)
                maj_indexes.append(idx)

        min_indexes = np.array(min_indexes)
        maj_indexes = np.array(maj_indexes)
        min_count = len(min_indexes)
        maj_count = len(maj_indexes)

        current_corr = 100 * maj_count / (maj_count + min_count)
        print(f"[Before subsampling] Current spurious_correlation: {current_corr:.2f}%, Target: {spurious_correlation}%")

        # No subsampling needed
        if spurious_correlation is None or abs(current_corr - spurious_correlation) < 0.5:
            return

        for i in range(100):  # try to find the best sampling match
            if spurious_correlation == 100:
                chosen_maj = maj_indexes
                chosen_min = np.array([], dtype=int)
            elif spurious_correlation == 0:
                chosen_min = min_indexes
                chosen_maj = np.array([], dtype=int)
            elif current_corr > spurious_correlation:
                # Too many maj → downsample majority
                if (spurious_correlation * (min_count - i)) % (100 - spurious_correlation) == 0:
                    b = min_count - i
                    a = (spurious_correlation * b) // (100 - spurious_correlation)
                    chosen_min = min_indexes[:b]
                    chosen_maj = maj_indexes[:a]
                    break
            else:
                # Too few maj → downsample minority
                if ((100 - spurious_correlation) * (maj_count - i)) % spurious_correlation == 0:
                    a = maj_count - i
                    b = ((100 - spurious_correlation) * a) // spurious_correlation
                    chosen_min = min_indexes[:b]
                    chosen_maj = maj_indexes[:a]
                    break
        indexes = np.concatenate([chosen_min, chosen_maj])
        print(f"[After subsampling] Maj: {len(chosen_maj)}, Min: {len(chosen_min)}, spurious_correlation: {100 * len(chosen_maj)/(len(indexes)):.2f}%")
        
        # Update data
        self.y_array = self.y_array[indexes]
        self.gender_array = self.gender_array[indexes]
        self.filename_array = self.filename_array[indexes]
        self.image_id_array = self.image_id_array[indexes]
            

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        return img, y, self.env_dict[(y, gender)], gender
