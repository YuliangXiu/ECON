# ignore all the warnings
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger("wandb").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

import numpy as np
import torch
import argparse
import os.path as osp
import os
from lib.common.train_util import SubTrainer, rename
from lib.common.config import get_cfg_defaults
from lib.dataset.PIFuDataModule import PIFuDataModule
from apps.ICON import ICON
from termcolor import colored
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config_file", type=str, help="path of the yaml config file"
    )
    parser.add_argument("-test", "--test_mode", action="store_true")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.ckpt_dir, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.results_path, "wandb"), exist_ok=True)

    os.environ["WANDB_NOTEBOOK_NAME"] = osp.join(cfg.results_path, f"wandb")
    wandb_logger = pl_loggers.WandbLogger(
        project="ICON", save_dir=cfg.results_path, name=f"{cfg.name}-{'-'.join(cfg.dataset.types)}"
    )

    if cfg.overfit:
        cfg_overfit_list = ["batch_size", 1]
        cfg.merge_from_list(cfg_overfit_list)
        save_k = 0

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=3,
        verbose=False,
        save_weights_only=True,
        monitor="val/avgloss",
        mode="min",
        filename="{epoch:02d}",
    )

    if cfg.test_mode or args.test_mode:

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
        cfg.merge_from_list(cfg_test_mode)

    # customized progress_bar
    theme = RichProgressBarTheme(description="green_yellow",
                                 progress_bar="green1",
                                 metrics="grey82")
    progress_bar = RichProgressBar(theme=theme)

    profiler = AdvancedProfiler(dirpath=osp.join(cfg.results_path, cfg.name),
                                filename="perf_logs")

    trainer_kwargs = {
        "accelerator": 'gpu',
        "devices": 1,
        "reload_dataloaders_every_n_epochs": 1,
        "sync_batchnorm": True,
        "benchmark": True,
        "profiler": profiler,
        "logger": wandb_logger,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [LearningRateMonitor(logging_interval="step"), checkpoint, progress_bar],
    }

    datamodule = PIFuDataModule(cfg)

    if not cfg.test_mode:
        datamodule.setup(stage="fit")
        train_len = datamodule.data_size["train"]
        val_len = datamodule.data_size["val"]
        trainer_kwargs.update(
            {
                "log_every_n_steps": int(cfg.freq_plot * train_len // cfg.batch_size),
                "val_check_interval": int(cfg.freq_eval * train_len / cfg.batch_size)
            }
        )

        if cfg.overfit:
            cfg_show_list = ["freq_show_train", 100.0, "freq_show_val", 10.0]
        else:
            cfg_show_list = [
                "freq_show_train", cfg.freq_show_train * train_len // cfg.batch_size,
                "freq_show_val", max(cfg.freq_show_val * val_len, 1.0),
            ]

        cfg.merge_from_list(cfg_show_list)

    model = ICON(cfg)

    trainer = SubTrainer(**trainer_kwargs)

    if (
        cfg.resume
        and os.path.exists(cfg.resume_path)
        and cfg.resume_path.endswith("ckpt")
    ):

        trainer_kwargs["resume_from_checkpoint"] = cfg.resume_path
        trainer = SubTrainer(**trainer_kwargs)
        print(
            colored(f"Resume weights and hparams from {cfg.resume_path}", "green"))

    elif not cfg.resume:

        model_dict = model.state_dict()
        main_dict = {}
        normal_dict = {}

        if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
            main_dict = torch.load(
                cfg.resume_path, map_location=torch.device(
                    f"cuda:{cfg.gpus[0]}")
            )["state_dict"]

            main_dict = {
                k: v
                for k, v in main_dict.items()
                if k in model_dict
                and v.shape == model_dict[k].shape
                and ("reconEngine" not in k)
                and ("normal_filter" not in k)
                and ("voxelization" not in k)
            }
            print(
                colored(f"Resume MLP weights from {cfg.resume_path}", "green"))

        if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
            normal_dict = torch.load(
                cfg.normal_path, map_location=torch.device(
                    f"cuda:{cfg.gpus[0]}")
            )["state_dict"]

            for key in normal_dict.keys():
                normal_dict = rename(
                    normal_dict, key, key.replace("netG", "netG.normal_filter")
                )

            normal_dict = {
                k: v
                for k, v in normal_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            print(
                colored(f"Resume normal model from {cfg.normal_path}", "green"))

        model_dict.update(main_dict)
        model_dict.update(normal_dict)
        model.load_state_dict(model_dict)

        del main_dict
        del normal_dict
        del model_dict

    else:
        pass

    if not cfg.test_mode:
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
    else:
        np.random.seed(1993)
        trainer.test(model=model, datamodule=datamodule)
