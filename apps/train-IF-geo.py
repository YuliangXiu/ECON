# ignore all the warnings
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("wandb").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os.path as osp
import os
import torch
from apps.IFGeo import IFGeo
from lib.common.train_util import SubTrainer, load_networks
from lib.common.config import get_cfg_defaults
from lib.dataset.IFDataModule import IFDataModule, cfg_test_list, cfg_overfit_list
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPSpawnStrategy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, help="path of the yaml config file")
    parser.add_argument("-test", "--test_mode", action="store_true")
    parser.add_argument("-overfit", "--overfit_mode", action="store_true")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    cfg.overfit = args.overfit_mode
    cfg.test_mode = args.test_mode
    cfg.freeze()

    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.ckpt_dir, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.results_path, "wandb"), exist_ok=True)

    wandb_logger = pl_loggers.WandbLogger(
        offline=False,
        project="IF-Geo",
        save_dir=osp.join(cfg.results_path, "wandb"),
        name=f"{cfg.name}-{cfg.dataset.voxel_res}-{'-'.join(cfg.dataset.types)}",
    )

    if cfg.overfit:
        cfg.merge_from_list(cfg_overfit_list)

    if cfg.test_mode:
        cfg.merge_from_list(cfg_test_list)

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=False,
        save_weights_only=True,
        monitor="val/avgloss",
        mode="min",
        filename="epoch={epoch:02d}-val_avgloss={val/avgloss:.2f}",
    )

    # customized progress_bar
    theme = RichProgressBarTheme(description="green_yellow",
                                 progress_bar="green1",
                                 metrics="grey82")
    progress_bar = RichProgressBar(theme=theme)

    trainer_kwargs = {
        "accelerator": "gpu",
        "devices": cfg.devices,
        "strategy": DDPSpawnStrategy(find_unused_parameters=False),
        "reload_dataloaders_every_n_epochs": 0,
        "sync_batchnorm": True,
        "benchmark": True,
        "logger": wandb_logger,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [
            LearningRateMonitor(logging_interval="step"),
            checkpoint,
            progress_bar,
        ],
    }

    datamodule = IFDataModule(cfg)

    if not cfg.test_mode:
        datamodule.prepare_data()
        train_len = datamodule.data_size["train"]
        val_len = datamodule.data_size["val"]
        trainer_kwargs.update({
            "log_every_n_steps": int(cfg.freq_plot * train_len // cfg.batch_size // cfg.devices),
            "val_check_interval": int(cfg.freq_eval * train_len // cfg.batch_size // cfg.devices),
        })

        if cfg.overfit:
            cfg_show_list = ["freq_show_train", 10.0, "freq_show_val", 10.0]
        else:
            cfg_show_list = [
                "freq_show_train",
                cfg.freq_show_train * train_len // cfg.batch_size // cfg.devices,
                "freq_show_val",
                max(cfg.freq_show_val * val_len // cfg.batch_size // cfg.devices, 1.0),
            ]

        cfg.merge_from_list(cfg_show_list)

    trainer = SubTrainer(**trainer_kwargs)
    model = IFGeo(cfg)

    # load checkpoints
    load_networks(model, mlp_path=cfg.resume_path)

    if not cfg.test_mode:
        trainer.fit(model=model, datamodule=datamodule)
