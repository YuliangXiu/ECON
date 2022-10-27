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

from lib.common.seg3d_lossless import Seg3dLossless
from lib.net.IFGeoNet import IFGeoNet
from lib.common.train_util import *
import torch
import wandb
import numpy as np
from skimage.transform import resize
import pytorch_lightning as pl

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

        self.netG = IFGeoNet(cfg)

        self.resolutions = (np.logspace(
            start=5,
            stop=np.log2(self.mcube_res),
            base=2,
            num=int(np.log2(self.mcube_res) - 4),
            endpoint=True,
        ) + 1.0)

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

            optimizer_G = torch.optim.Adadelta(optim_params_G,
                                               lr=self.lr_G,
                                               weight_decay=weight_decay)

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
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G,
                                                           milestones=self.cfg.schedule,
                                                           gamma=self.cfg.gamma)

        return [optimizer_G], [scheduler_G]

    def training_step(self, batch, batch_idx):

        # cfg log
        if self.cfg.devices == 1:
            if not self.cfg.fast_dev and self.global_step == 0:
                export_cfg(self.logger, osp.join(self.cfg.results_path, self.cfg.name), self.cfg)
                self.logger.experiment.config.update(convert_to_dict(self.cfg))

        self.netG.train()

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        # metrics processing
        metrics_log = {
            "loss": error_G,
        }

        self.log_dict(metrics_log,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=False,
                      sync_dist=True)

        if self.cfg.devices == 1:
            if batch_idx % int(self.cfg.freq_show_train) == 0:

                with torch.no_grad():
                    self.render_func(batch, dataset="train", idx=self.global_step)

        return metrics_log

    def training_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "train/avgloss": batch_mean(outputs, "loss"),
        }

        self.log_dict(metrics_log,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      rank_zero_only=True)

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        if self.cfg.devices == 1:
            if batch_idx % int(self.cfg.freq_show_val) == 0:
                with torch.no_grad():
                    self.render_func(batch, dataset="val", idx=batch_idx)

        metrics_log = {
            "val/loss": error_G,
        }

        self.log_dict(metrics_log,
                      prog_bar=True,
                      logger=False,
                      on_step=True,
                      on_epoch=False,
                      sync_dist=True)

        return metrics_log

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val/avgloss": batch_mean(outputs, "val/loss"),
        }

        self.log_dict(metrics_log,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      rank_zero_only=True)


    def render_func(self, batch, dataset="title", idx=0):

        def resize_img(img, height):
            img = img.repeat(1, 3 // img.shape[1], 1, 1)
            resize_img = resize(
                ((img.cpu().numpy()[0] + 1.0) / 2.0 * 255.0).transpose(1, 2, 0),
                (height, height),
                anti_aliasing=True,
            )
            return resize_img

        for name in batch.keys():
            if batch[name] is not None:
                batch[name] = batch[name][0:1]

        self.netG.eval()
        sdf = self.reconEngine(netG=self.netG, batch=batch)

        if sdf is not None:
            render = self.reconEngine.display(sdf)

            image_pred = np.flip(render[:, :, ::-1], axis=0)
            height = image_pred.shape[0]

            image = PIL.Image.fromarray(
                np.concatenate(
                    [
                        image_pred,
                        np.concatenate([
                            resize_img(batch["image"], height),
                            resize_img(batch["depth_F"] * batch["depth_mask"], height),
                            resize_img(batch["depth_B"] * batch["depth_mask"], height),
                            resize_img(batch["image_back"], height)
                        ],
                                       axis=1),
                    ],
                    axis=0,
                ).astype(np.uint8))

            self.logger.log_image(key=f"SDF/{dataset}/{idx if not self.overfit else 1}",
                                  images=[wandb.Image(image, caption="multi-views")])
