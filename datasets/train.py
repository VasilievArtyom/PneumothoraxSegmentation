import os

import cv2
import pydicom
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.transforms import preprocess, transforms


class TrainDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.annotations = []
        self.mode = mode
        self.img_dir = os.path.join(
            config["data_path"],
            self.config['images'] if (mode == 'train') else self.config['val_images']
        )
        self.masks_dir = os.path.join(
            config["data_path"],
            self.config['masks'] if (mode == 'train') else self.config['val_masks']
        )
        csv_path = os.path.join(
            config["annotation_path"],
            self.config['annotations'] if (mode == 'train') else self.config['val_annotations']
        )
        self._read_csv(csv_path)
        self._init_transforms()

    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            self.annotations.append(
                [
                    row['sop_path'],
                    row['mask_path'],
                    int(row['PNTX'])
                ]
            )

    def _init_transforms(self):
        self.preprocess = preprocess(self.config['preprocess'])
        self.transforms = None if self.mode != "train" else transforms(self.config['transforms'])

    def __len__(self):
        return len(self.annotations)

    def load_sample(self, idx):
        image_path, mask_path, label = self.annotations[idx]
        if not os.path.exists(os.path.join(self.img_dir, image_path)):
            raise ValueError(f"dcm {os.path.join(self.img_dir, image_path)} doesn't exist")
        
        if not os.path.exists(os.path.join(self.masks_dir, mask_path)):
            raise ValueError(f"mask {os.path.join(self.masks_dir, mask_path)} doesn't exist")
        
        image = pydicom.dcmread(
            os.path.join(self.img_dir, image_path)
        ).pixel_array

        image = np.stack([image,image,image], axis=2)

        mask = cv2.imread(
            os.path.join(self.masks_dir, mask_path),
            cv2.IMREAD_GRAYSCALE
        )
        return image, mask, label

    def __getitem__(self, idx):
        image, mask, label = self.load_sample(idx)
        mask = mask / 255
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                mask=mask
            )
            image = transformed['image']
            mask = transformed['mask']
        
        preprocessed = self.preprocess(
            image=image,
            mask=mask
        )
        image = preprocessed['image']
        mask = preprocessed['mask'].unsqueeze(0).float()
        
        label = torch.as_tensor([label]).float()
        idx = torch.as_tensor(idx).long()
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'idx': idx
        }


def collate_fn(batch):
    image = torch.stack([b['image'] for b in batch], dim=0)
    mask = torch.stack([b['mask'] for b in batch], dim=0)
    label = torch.stack([b['label'] for b in batch], dim=0).reshape(-1)
    return image, mask, label


def get_train_dl_ds(
        config,
        mode='train'
):
    dataset = TrainDataset(
        config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
