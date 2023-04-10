import numpy as np
import pytorch_lightning as pl
import torch
from skimage.transform import resize

from lib.common.train_util import batch_mean
from lib.net import NormalNet


class Normal(pl.LightningModule):
    def __init__(self, cfg):
        super(Normal, self).__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_F = self.cfg.lr_netF
        self.lr_B = self.cfg.lr_netB
        self.lr_D = self.cfg.lr_netD
        self.overfit = cfg.overfit

        self.F_losses = [item[0] for item in self.cfg.net.front_losses]
        self.B_losses = [item[0] for item in self.cfg.net.back_losses]
        self.ALL_losses = self.F_losses + self.B_losses

        self.automatic_optimization = False

        self.schedulers = []

        self.netG = NormalNet(self.cfg)

        self.in_nml = [item[0] for item in cfg.net.in_nml]

    # Training related
    def configure_optimizers(self):

        optim_params_N_D = None
        optimizer_N_D = None
        scheduler_N_D = None

        # set optimizer
        optim_params_N_F = [{"params": self.netG.netF.parameters(), "lr": self.lr_F}]
        optim_params_N_B = [{"params": self.netG.netB.parameters(), "lr": self.lr_B}]

        optimizer_N_F = torch.optim.Adam(optim_params_N_F, lr=self.lr_F, betas=(0.5, 0.999))
        optimizer_N_B = torch.optim.Adam(optim_params_N_B, lr=self.lr_B, betas=(0.5, 0.999))

        scheduler_N_F = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_F, milestones=self.cfg.schedule, gamma=self.cfg.gamma
        )

        scheduler_N_B = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_B, milestones=self.cfg.schedule, gamma=self.cfg.gamma
        )
        if 'gan' in self.ALL_losses:
            optim_params_N_D = [{"params": self.netG.netD.parameters(), "lr": self.lr_D}]
            optimizer_N_D = torch.optim.Adam(optim_params_N_D, lr=self.lr_D, betas=(0.5, 0.999))
            scheduler_N_D = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_N_D, milestones=self.cfg.schedule, gamma=self.cfg.gamma
            )
            self.schedulers = [scheduler_N_F, scheduler_N_B, scheduler_N_D]
            optims = [optimizer_N_F, optimizer_N_B, optimizer_N_D]

        else:
            self.schedulers = [scheduler_N_F, scheduler_N_B]
            optims = [optimizer_N_F, optimizer_N_B]

        return optims, self.schedulers

    def render_func(self, render_tensor, dataset, idx):

        height = render_tensor["image"].shape[2]
        result_list = []

        for name in render_tensor.keys():
            result_list.append(
                resize(
                    ((render_tensor[name].cpu().numpy()[0] + 1.0) / 2.0).transpose(1, 2, 0),
                    (height, height),
                    anti_aliasing=True,
                )
            )

        self.logger.log_image(
            key=f"Normal/{dataset}/{idx if not self.overfit else 1}",
            images=[(np.concatenate(result_list, axis=1) * 255.0).astype(np.uint8)]
        )

    def training_step(self, batch, batch_idx):

        self.netG.train()

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {"normal_F": batch["normal_F"], "normal_B": batch["normal_B"]}

        in_tensor.update(FB_tensor)

        preds_F, preds_B = self.netG(in_tensor)
        error_dict = self.netG.get_norm_error(preds_F, preds_B, FB_tensor)

        if 'gan' in self.ALL_losses:
            (opt_F, opt_B, opt_D) = self.optimizers()
            opt_F.zero_grad()
            self.manual_backward(error_dict["netF"])
            opt_B.zero_grad()
            self.manual_backward(error_dict["netB"], retain_graph=True)
            opt_D.zero_grad()
            self.manual_backward(error_dict["netD"])
            opt_F.step()
            opt_B.step()
            opt_D.step()
        else:
            (opt_F, opt_B) = self.optimizers()
            opt_F.zero_grad()
            self.manual_backward(error_dict["netF"])
            opt_B.zero_grad()
            self.manual_backward(error_dict["netB"])
            opt_F.step()
            opt_B.step()

        if batch_idx > 0 and batch_idx % int(
            self.cfg.freq_show_train
        ) == 0 and self.cfg.devices == 1:

            self.netG.eval()
            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                self.render_func(in_tensor, "train", self.global_step)

        # metrics processing
        metrics_log = {"loss": error_dict["netF"] + error_dict["netB"]}

        if "gan" in self.ALL_losses:
            metrics_log["loss"] += error_dict["netD"]

        for key in error_dict.keys():
            metrics_log["train/loss_" + key] = error_dict[key].item()

        self.log_dict(
            metrics_log, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True
        )

        return metrics_log

    def training_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {}
        for key in outputs[0].keys():
            if "/" in key:
                [stage, loss_name] = key.split("/")
            else:
                stage = "train"
                loss_name = key
            metrics_log[f"{stage}/avg-{loss_name}"] = batch_mean(outputs, key)

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

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {"normal_F": batch["normal_F"], "normal_B": batch["normal_B"]}
        in_tensor.update(FB_tensor)

        preds_F, preds_B = self.netG(in_tensor)
        error_dict = self.netG.get_norm_error(preds_F, preds_B, FB_tensor)

        if batch_idx % int(self.cfg.freq_show_train) == 0 and self.cfg.devices == 1:

            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                self.render_func(in_tensor, "val", batch_idx)

        # metrics processing
        metrics_log = {"val/loss": error_dict["netF"] + error_dict["netB"]}

        if "gan" in self.ALL_losses:
            metrics_log["val/loss"] += error_dict["netD"]

        for key in error_dict.keys():
            metrics_log["val/" + key] = error_dict[key].item()

        return metrics_log

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {}
        for key in outputs[0].keys():
            [stage, loss_name] = key.split("/")
            metrics_log[f"{stage}/avg-{loss_name}"] = batch_mean(outputs, key)

        self.log_dict(
            metrics_log,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True
        )
