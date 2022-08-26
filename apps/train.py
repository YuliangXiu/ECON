# ignore all the warnings
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger("wandb").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)


from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from apps.ICON import ICON
from lib.dataset.PIFuDataModule import PIFuDataModule
from lib.common.config import get_cfg_defaults
from lib.common.train_util import SubTrainer, load_networks
import os
import os.path as osp
import argparse
import numpy as np

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

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.results_path, name=cfg.name, default_hp_metric=False
    )

    if cfg.overfit:
        cfg_overfit_list = ["batch_size", 1]
        cfg.merge_from_list(cfg_overfit_list)
        save_k = 0

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=1,
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

    freq_eval = cfg.freq_eval
    if cfg.fast_dev > 0:
        freq_eval = cfg.fast_dev

    trainer_kwargs = {
        "gpus": cfg.gpus,
        "auto_select_gpus": True,
        "reload_dataloaders_every_epoch": True,
        "sync_batchnorm": True,
        "benchmark": True,
        "logger": tb_logger,
        "track_grad_norm": -1,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "checkpoint_callback": checkpoint,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "profiler": None,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [LearningRateMonitor(logging_interval="step")],
    }

    datamodule = PIFuDataModule(cfg)

    if not cfg.test_mode:
        datamodule.setup(stage="fit")
        train_len = datamodule.data_size["train"]
        val_len = datamodule.data_size["val"]
        trainer_kwargs.update(
            {
                "log_every_n_steps": int(cfg.freq_plot * train_len // cfg.batch_size),
                "val_check_interval": int(freq_eval * train_len // cfg.batch_size)
                if freq_eval > 10
                else freq_eval,
            }
        )

        if cfg.overfit:
            cfg_show_list = ["freq_show_train", 100.0, "freq_show_val", 10.0]
        else:
            cfg_show_list = [
                "freq_show_train",
                cfg.freq_show_train * train_len // cfg.batch_size,
                "freq_show_val",
                max(cfg.freq_show_val * val_len, 1.0),
            ]

        cfg.merge_from_list(cfg_show_list)

    model = ICON(cfg)

    trainer = SubTrainer(**trainer_kwargs)

    # load checkpoints
    load_networks(cfg, model,
                  mlp_path=cfg.resume_path,
                  normal_path=cfg.normal_path)

    if not cfg.test_mode:
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
    else:
        np.random.seed(1993)
        trainer.test(model=model, datamodule=datamodule)
