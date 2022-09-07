from torch.utils.data import DataLoader
from .PIFuDataset import PIFuDataset
import pytorch_lightning as pl

cfg_test_mode = [
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
]


class PIFuDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(PIFuDataModule, self).__init__()
        self.cfg = cfg
        self.overfit = self.cfg.overfit

        if self.overfit:
            self.batch_size = 1
        else:
            self.batch_size = self.cfg.batch_size

        self.data_size = {}

    def prepare_data(self):

        pass

    def setup(self, stage):

        if stage == 'fit':
            self.train_dataset = PIFuDataset(cfg=self.cfg, split="train")
            self.val_dataset = PIFuDataset(cfg=self.cfg, split="val")
            self.data_size = {'train': len(self.train_dataset),
                              'val': len(self.val_dataset)}

        if stage == 'test':
            self.cfg.merge_from_list(cfg_test_mode)
            self.test_dataset = PIFuDataset(cfg=self.cfg, split="test")

    def train_dataloader(self):

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.cfg.num_threads, pin_memory=True)

        return train_data_loader

    def val_dataloader(self):

        if self.overfit:
            current_dataset = self.train_dataset
        else:
            current_dataset = self.val_dataset

        val_data_loader = DataLoader(
            current_dataset,
            batch_size=1, shuffle=False,
            num_workers=self.cfg.num_threads, pin_memory=True)

        return val_data_loader

    def test_dataloader(self):

        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1, shuffle=False,
            num_workers=self.cfg.num_threads, pin_memory=True)

        return test_data_loader
