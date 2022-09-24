from lib.net import NormalNet
from lib.common.train_util import convert_to_dict, export_cfg, batch_mean
import torch
import numpy as np
import os.path as osp
import wandb
from torch import nn
from skimage.transform import resize
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True


class Normal(pl.LightningModule):

    def __init__(self, cfg):
        super(Normal, self).__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_N = self.cfg.lr_N
        self.overfit = cfg.overfit

        self.automatic_optimization = False

        self.schedulers = []

        self.netG = NormalNet(self.cfg, error_term=nn.SmoothL1Loss())

        self.in_nml = [item[0] for item in cfg.net.in_nml]

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay

        optim_params_N_F = [{
            "params": self.netG.netF.parameters(),
            "lr": self.lr_N
        }]
        optim_params_N_B = [{
            "params": self.netG.netB.parameters(),
            "lr": self.lr_N
        }]

        optimizer_N_F = torch.optim.Adam(optim_params_N_F,
                                         lr=self.lr_N,
                                         weight_decay=weight_decay)

        optimizer_N_B = torch.optim.Adam(optim_params_N_B,
                                         lr=self.lr_N,
                                         weight_decay=weight_decay)

        scheduler_N_F = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_F, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        scheduler_N_B = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_B, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        self.schedulers = [scheduler_N_F, scheduler_N_B]
        optims = [optimizer_N_F, optimizer_N_B]

        return optims, self.schedulers

    def render_func(self, render_tensor, dataset, idx):

        height = render_tensor["image"].shape[2]
        result_list = []

        for name in render_tensor.keys():
            result_list.append(
                resize(
                    ((render_tensor[name].cpu().numpy()[0] + 1.0) /
                     2.0).transpose(1, 2, 0),
                    (height, height),
                    anti_aliasing=True,
                ))
        result_array = np.concatenate(result_list, axis=1)

        self.logger.experiment.log({
            f"Normal/{dataset}/{idx if not self.overfit else 1}":
            wandb.Image(result_array)
        })

    def training_step(self, batch, batch_idx):

        # cfg log
        if not self.cfg.fast_dev and self.global_step == 0:
            export_cfg(self.logger,
                       osp.join(self.cfg.results_path, self.cfg.name),
                       self.cfg)
            self.logger.experiment.config.update(convert_to_dict(self.cfg))

        self.netG.train()

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {
            "normal_F": batch["normal_F"],
            "normal_B": batch["normal_B"]
        }

        in_tensor.update(FB_tensor)

        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B,
                                                      FB_tensor)

        (opt_nf, opt_nb) = self.optimizers()

        opt_nf.zero_grad()
        self.manual_backward(error_NF)
        opt_nf.step()

        opt_nb.zero_grad()
        self.manual_backward(error_NB)
        opt_nb.step()

        if batch_idx > 0 and batch_idx % int(self.cfg.freq_show_train) == 0:

            self.netG.eval()
            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                self.render_func(in_tensor, "train", self.global_step)

        # metrics processing
        metrics_log = {
            "loss": 0.5 * (error_NF + error_NB),
            "train/loss-NF": error_NF.item(),
            "train/loss-NB": error_NB.item(),
        }

        self.log_dict(metrics_log,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=False)

        return metrics_log

    def training_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "train/avgloss": batch_mean(outputs, "loss"),
            "train/avgloss-NF": batch_mean(outputs, "train/loss-NF"),
            "train/avgloss-NB": batch_mean(outputs, "train/loss-NB"),
        }

        self.log_dict(metrics_log,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {
            "normal_F": batch["normal_F"],
            "normal_B": batch["normal_B"]
        }
        in_tensor.update(FB_tensor)

        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B,
                                                      FB_tensor)

        if batch_idx % int(self.cfg.freq_show_train) == 0:

            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                self.render_func(in_tensor, "val", batch_idx)

        metrics_log = {
            "val/loss": 0.5 * (error_NF + error_NB),
            "val/loss-NF": error_NF,
            "val/loss-NB": error_NB,
        }

        return metrics_log

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val/avgloss": batch_mean(outputs, "val/loss"),
            "val/avgloss-NF": batch_mean(outputs, "val/loss-NF"),
            "val/avgloss-NB": batch_mean(outputs, "val/loss-NB"),
        }

        self.log_dict(metrics_log,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)
