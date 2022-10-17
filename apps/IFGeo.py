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
from lib.dataset.Evaluator import Evaluator
from lib.net.IFGeoNet import IFGeoNet
from lib.common.train_util import *
from lib.common.render import Render
import torch
import wandb
import numpy as np
from skimage.transform import resize
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True


class IFGeo(pl.LightningModule):

    def __init__(self, cfg, device):
        super(IFGeo, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_G = self.cfg.lr_G

        self.use_sdf = cfg.sdf
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh
        self.overfit = cfg.overfit

        self.netG = IFGeoNet(cfg, device)

        self.evaluator = Evaluator(device)

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
            device=device,
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=True,
        )

        self.render = Render(size=512, device=device)

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
        if not self.cfg.fast_dev and self.global_step == 0:
            export_cfg(self.logger, osp.join(self.cfg.results_path, self.cfg.name), self.cfg)
            self.logger.experiment.config.update(convert_to_dict(self.cfg))

        self.netG.train()

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            batch["labels_geo"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        # metrics processing
        metrics_log = {
            "loss": error_G,
            "train/acc": acc.item(),
            "train/iou": iou.item(),
            "train/prec": prec.item(),
            "train/recall": recall.item(),
        }

        self.log_dict(metrics_log, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if batch_idx % int(self.cfg.freq_show_train) == 0:

            with torch.no_grad():
                self.render_func(batch, dataset="train", idx=self.global_step)

        return metrics_log

    def training_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "train/avgloss": batch_mean(outputs, "loss"),
            "train/avgiou": batch_mean(outputs, "train/iou"),
            "train/avgprec": batch_mean(outputs, "train/prec"),
            "train/avgrecall": batch_mean(outputs, "train/recall"),
            "train/avgacc": batch_mean(outputs, "train/acc"),
        }

        self.log_dict(metrics_log, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        preds_G = self.netG(batch)
        error_G = self.netG.compute_loss(preds_G, batch["labels_geo"])

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            batch["labels_geo"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        if batch_idx % int(self.cfg.freq_show_val) == 0:
            with torch.no_grad():
                self.render_func(batch, dataset="val", idx=batch_idx)

        metrics_log = {
            "val/loss": error_G,
            "val/acc": acc,
            "val/iou": iou,
            "val/prec": prec,
            "val/recall": recall,
        }

        self.log_dict(metrics_log, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        return metrics_log

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val/avgloss": batch_mean(outputs, "val/loss"),
            "val/avgacc": batch_mean(outputs, "val/acc"),
            "val/avgiou": batch_mean(outputs, "val/iou"),
            "val/avgprec": batch_mean(outputs, "val/prec"),
            "val/avgrecall": batch_mean(outputs, "val/recall"),
        }

        self.log_dict(metrics_log, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False
        
        # export paths
        mesh_name = batch["subject"][0]
        mesh_rot = batch["rotation"][0].item()

        self.export_dir = osp.join(
            self.cfg.results_path,
            self.cfg.name,
            "-".join(self.cfg.dataset.types),
            mesh_name,
        )
        os.makedirs(self.export_dir, exist_ok=True)

        with torch.no_grad():
            sdf = self.reconEngine(netG=self.netG, batch=batch)

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_gt = batch["verts"][0]
        faces_gt = batch["faces"][0]

        self.result_eval.update({
            "verts_gt": verts_gt,
            "faces_gt": faces_gt,
            "verts_pr": verts_pr,
            "faces_pr": faces_pr,
            "recon_size": (self.resolutions[-1] - 1.0),
            "calib": batch["calib"][0],
        })

        self.evaluator.set_mesh(self.result_eval)
        chamfer, p2s = self.evaluator.calculate_chamfer_p2s(num_samples=1000)
        nc = self.evaluator.calculate_normal_consist(osp.join(self.export_dir,
                                                              f"{mesh_rot}_nc.png"))

        test_log = {"chamfer": chamfer, "p2s": p2s, "NC": nc}

        self.log_dict(test_log, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        return test_log

    def test_epoch_end(self, outputs):

        accu_outputs = accumulate(
            outputs,
            rot_num=3,
            split={
                "cape-easy": (0, 50),
                "cape-hard": (50, 100)
            },
        )

        print(colored(self.cfg.name, "green"))
        print(colored(self.cfg.dataset.noise_scale, "green"))

        self.log_dict(accu_outputs, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        np.save(
            osp.join(self.export_dir, "../test_results.npy"),
            accu_outputs,
            allow_pickle=True,
        )

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
                            resize_img(batch["depth_F"], height),
                            resize_img(batch["depth_B"], height),
                            resize_img(batch["image_back"], height)
                        ],
                                       axis=1),
                    ],
                    axis=0,
                ).astype(np.uint8))

            self.logger.log_image(key=f"SDF/{dataset}/{idx if not self.overfit else 1}",
                                  images=[wandb.Image(image, caption="multi-views")])
