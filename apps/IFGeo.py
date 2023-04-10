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

import numpy as np
import pytorch_lightning as pl
import torch

from lib.common.seg3d_lossless import Seg3dLossless
from lib.common.train_util import *

torch.backends.cudnn.benchmark = True


class IFGeo(pl.LightningModule):
    def __init__(self, cfg):
        super(IFGeo, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_G = self.cfg.lr_G

        self.use_sdf = cfg.sdf
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh
        self.overfit = cfg.overfit

        if cfg.dataset.prior_type == "SMPL":
            from lib.net.IFGeoNet import IFGeoNet
            self.netG = IFGeoNet(cfg)
        else:
            from lib.net.IFGeoNet_nobody import IFGeoNet
            self.netG = IFGeoNet(cfg)

        self.resolutions = (
            np.logspace(
                start=5,
                stop=np.log2(self.mcube_res),
                base=2,
                num=int(np.log2(self.mcube_res) - 4),
                endpoint=True,
            ) + 1.0
        )

        self.resolutions = self.resolutions.astype(np.int16).tolist()

        self.reconEngine = Seg3dLossless(
            query_func=query_func_IF,
            b_min=[[-1.0, 1.0, -1.0]],
            b_max=[[1.0, -1.0, 1.0]],
            resolutions=self.resolutions,
            align_corners=True,
            balance_value=0.50,
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=True,
        )

        self.export_dir = None
        self.result_eval = {}

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum

        optim_params_G = [{"params": self.netG.parameters(), "lr": self.lr_G}]

        if self.cfg.optim == "Adadelta":

            optimizer_G = torch.optim.Adadelta(
                optim_params_G, lr=self.lr_G, weight_decay=weight_decay
            )

        elif self.cfg.optim == "Adam":

            optimizer_G = torch.optim.Adam(optim_params_G, lr=self.lr_G, weight_decay=weight_decay)

        elif self.cfg.optim == "RMSprop":

            optimizer_G = torch.optim.RMSprop(
                optim_params_G,
                lr=self.lr_G,
                weight_decay=weight_decay,
                momentum=momentum,
            )

        else:
            raise NotImplementedError

        # set scheduler
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G, milestones=self.cfg.schedule, gamma=self.cfg.gamma
        )

        return [optimizer_G], [scheduler_G]

    def training_step(self, batch, batch_idx):

        self.netG.train()

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        # metrics processing
        metrics_log = {
            "loss": error_G,
        }

        self.log_dict(
            metrics_log, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True
        )

        return metrics_log

    def training_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "train/avgloss": batch_mean(outputs, "loss"),
        }

        self.log_dict(
            metrics_log,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True
        )

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        metrics_log = {
            "val/loss": error_G,
        }

        self.log_dict(
            metrics_log, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True
        )

        return metrics_log

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val/avgloss": batch_mean(outputs, "val/loss"),
        }

        self.log_dict(
            metrics_log,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True
        )
