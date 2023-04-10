# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# pytorch lightning related libs
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.dataset.NormalDataset import NormalDataset


class NormalModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(NormalModule, self).__init__()
        self.cfg = cfg

        self.batch_size = self.cfg.batch_size

        self.data_size = {}

    def prepare_data(self):

        pass

    def setup(self, stage):

        self.train_dataset = NormalDataset(cfg=self.cfg, split="train")
        self.val_dataset = NormalDataset(cfg=self.cfg, split="val")
        self.test_dataset = NormalDataset(cfg=self.cfg, split="test")

        self.data_size = {
            "train": len(self.train_dataset),
            "val": len(self.val_dataset),
        }

    def train_dataloader(self):

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_threads,
            pin_memory=True,
        )

        return train_data_loader

    def val_dataloader(self):

        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_threads,
            pin_memory=True,
        )

        return val_data_loader

    def val_dataloader(self):

        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_threads,
            pin_memory=True,
        )

        return test_data_loader
