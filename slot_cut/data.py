import json
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from slot_cut.utils import compact
from slot_cut.clevertex import CLEVRTEX


class CLEVRTEXDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str,
            train_batch_size: int,
            val_batch_size: int,
            clevr_transforms: Callable,
            max_n_objects: int,
            num_workers: int,
            resolution=(32,32),
            num_train_images: Optional[int] = None,
            num_val_images: Optional[int] = None,
            clevrtex_variant: str = "full",
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = CLEVRTEX(
            data_root,
            dataset_variant=clevrtex_variant,
            # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
            split='train',
            crop=True,
            resize=(128, 128),
            return_metadata = False  # Useful only for evaluation, wastes time on I/O otherwise
        )

        self.val_dataset = CLEVRTEX(
            data_root,
            dataset_variant=clevrtex_variant,
            # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
            split='val',
            crop=True,
            resize=(128, 128),
            return_metadata = False  # Useful only for evaluation, wastes time on I/O otherwise
        )

        self.test_dataset = CLEVRTEX(
            data_root,
            dataset_variant=clevrtex_variant,
            # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
            split='test',
            crop=True,
            resize=(128, 128),
            return_metadata = False  # Useful only for evaluation, wastes time on I/O otherwise
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

