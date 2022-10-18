from torch.utils.data import DataLoader
from .IFDataset import IFDataset
import pytorch_lightning as pl
import os.path as osp
import numpy as np

cfg_test_list = [
    "test_mode",
    True,
    "dataset.types",
    ["cape"],
    "dataset.scales",
    [100.0],
    "dataset.rotation_num",
    3,
    "mcube_res",
    256,
    "clean_mesh",
    True,
    "batch_size",
    1,
]

cfg_overfit_list = [
    "batch_size",
    1,
    "num_threads",
    1,
    "mcube_res",
    128,
    "freq_plot",
    0.0001,
]


class IFDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(IFDataModule, self).__init__()
        self.cfg = cfg
        self.overfit = self.cfg.overfit

        if self.overfit:
            self.batch_size = 1
        else:
            self.batch_size = self.cfg.batch_size

        self.data_size = {}

    def prepare_data(self):

        self.opt = self.cfg.dataset
        self.datasets = self.opt.types

        self.data_size = {"train": 0, "val": 0, "test": 0}

        for dataset in self.datasets:

            dataset_dir = osp.join(self.cfg.root, dataset)
            for split in ["train", "val", "test"]:
                self.data_size[split] += len(
                    np.loadtxt(osp.join(dataset_dir, f"{split}.txt"),
                               dtype=str)) * self.opt.rotation_num

        print(self.data_size)

    def setup(self, stage):

        if stage == "fit":
            self.train_dataset = IFDataset(cfg=self.cfg, split="train")
            self.val_dataset = IFDataset(cfg=self.cfg, split="val")

        if stage == "test":
            self.cfg.merge_from_list(cfg_test_list)
            self.test_dataset = IFDataset(cfg=self.cfg, split="test")

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

        if self.overfit:
            current_dataset = self.train_dataset
        else:
            current_dataset = self.val_dataset

        val_data_loader = DataLoader(
            current_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_threads,
            pin_memory=True,
        )

        return val_data_loader

    def test_dataloader(self):

        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_threads,
            pin_memory=True,
        )

        return test_data_loader
