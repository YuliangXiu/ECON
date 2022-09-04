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
from lib.net import HGPIFuNet
from lib.common.train_util import *
from lib.common.render import Render
from lib.dataset.mesh_util import SMPLX
import torch
import wandb
import numpy as np
from torch import nn
from skimage.transform import resize
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True


class ICON(pl.LightningModule):
    def __init__(self, cfg):
        super(ICON, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_G = self.cfg.lr_G

        self.use_sdf = cfg.sdf
        self.prior_type = cfg.net.prior_type
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh

        self.netG = HGPIFuNet(
            self.cfg,
            self.cfg.projection_mode,
            error_term=nn.SmoothL1Loss() if self.use_sdf else nn.MSELoss(),
        )

        self.evaluator = Evaluator(
            device=torch.device(f"cuda:{self.cfg.gpus[0]}"))

        self.resolutions = (
            np.logspace(
                start=5,
                stop=np.log2(self.mcube_res),
                base=2,
                num=int(np.log2(self.mcube_res) - 4),
                endpoint=True,
            )
            + 1.0
        )
        self.resolutions = self.resolutions.astype(np.int16).tolist()

        self.base_keys = ["smpl_verts", "smpl_faces"]
        self.feat_names = self.cfg.net.smpl_feats
        
        self.icon_keys = self.base_keys + \
            [f"smpl_{feat_name}" for feat_name in self.feat_names]
        self.keypoint_keys = self.base_keys + \
            [f"smpl_{feat_name}" for feat_name in self.feat_names]
        self.pamir_keys = [
            "voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"
        ]
        self.pifu_keys = []

        self.reconEngine = Seg3dLossless(
            query_func=query_func,
            b_min=[[-1.0, 1.0, -1.0]],
            b_max=[[1.0, -1.0, 1.0]],
            resolutions=self.resolutions,
            align_corners=True,
            balance_value=0.50,
            device=torch.device(f"cuda:{self.cfg.test_gpus[0]}"),
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=True,
        )

        self.render = Render(
            size=512, device=torch.device(f"cuda:{self.cfg.test_gpus[0]}")
        )
        self.smpl_data = SMPLX()

        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_total = self.in_geo + self.in_nml
        self.smpl_dim = cfg.net.smpl_dim

        self.export_dir = None
        self.result_eval = {}

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum

        optim_params_G = [
            {"params": self.netG.if_regressor.parameters(), "lr": self.lr_G}
        ]

        if self.cfg.net.use_filter:
            optim_params_G.append(
                {"params": self.netG.F_filter.parameters(), "lr": self.lr_G}
            )

        if self.cfg.net.prior_type == "pamir":
            optim_params_G.append(
                {"params": self.netG.ve.parameters(), "lr": self.lr_G}
            )
            
        if self.cfg.optim == "Adadelta":

            optimizer_G = torch.optim.Adadelta(
                optim_params_G, lr=self.lr_G, weight_decay=weight_decay
            )

        elif self.cfg.optim == "Adam":

            optimizer_G = torch.optim.Adam(
                optim_params_G, lr=self.lr_G, weight_decay=weight_decay
            )

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

        # cfg log
        if not self.cfg.fast_dev and self.global_step == 0:
            export_cfg(self.logger, osp.join(
                self.cfg.results_path, self.cfg.name), self.cfg)
            self.logger.experiment.config.update(convert_to_dict(self.cfg))

        self.netG.train()

        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        preds_G, error_G = self.netG(in_tensor_dict)

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
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

        self.log_dict(metrics_log, prog_bar=True, logger=True,
                      on_step=True, on_epoch=False)

        if batch_idx % int(self.cfg.freq_show_train) == 0:

            with torch.no_grad():
                self.render_func(
                    in_tensor_dict, dataset="train", idx=self.global_step)

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

        self.log_dict(metrics_log, prog_bar=False, logger=True,
                      on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        preds_G, error_G = self.netG(in_tensor_dict)

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        if batch_idx % int(self.cfg.freq_show_val) == 0:
            with torch.no_grad():
                self.render_func(in_tensor_dict, dataset="val", idx=batch_idx)

        metrics_log = {
            "val/loss": error_G,
            "val/acc": acc,
            "val/iou": iou,
            "val/prec": prec,
            "val/recall": recall,
        }

        self.log_dict(metrics_log, prog_bar=True, logger=False,
                      on_step=True, on_epoch=False)

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

        self.log_dict(metrics_log, prog_bar=False, logger=True,
                      on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        # dict_keys(['dataset', 'subject', 'rotation', 'scale', 'calib',
        #            'normal_F', 'normal_B', 'image', 'T_normal_F', 'T_normal_B',
        #            'z-trans', 'verts', 'faces', 'samples_geo', 'labels_geo',
        #            'smpl_verts', 'smpl_faces', 'smpl_vis', 'smpl_cmap',
        #            'type', 'gender', 'age', 'body_pose', 'global_orient', 'betas', 'transl'])

        self.netG.eval()
        self.netG.training = False
        in_tensor_dict = {}

        # export paths
        mesh_name = batch["subject"][0]
        mesh_rot = batch["rotation"][0].item()

        self.export_dir = osp.join(
            self.cfg.results_path, self.cfg.name, "-".join(self.cfg.dataset.types), mesh_name)
        os.makedirs(self.export_dir, exist_ok=True)

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        if "T_normal_F" not in in_tensor_dict.keys() or "T_normal_B" not in in_tensor_dict.keys():

            # update the new T_normal_F/B
            self.render.load_meshes(batch["smpl_verts"] * torch.tensor([1.0, -1.0, 1.0]).to(self.device),
                                    batch["smpl_faces"])
            T_normal_F, T_noraml_B = self.render.get_rgb_image()
            in_tensor_dict.update(
                {'T_normal_F': T_normal_F, 'T_normal_B': T_noraml_B})

        with torch.no_grad():
            features, inter = self.netG.filter(
                in_tensor_dict, return_inter=True)
            sdf = self.reconEngine(
                opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
            )

        def tensor2arr(x): return (x[0].permute(
            1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 * 255.0

        # save inter results
        image = tensor2arr(in_tensor_dict["image"])
        smpl_F = tensor2arr(in_tensor_dict["T_normal_F"])
        smpl_B = tensor2arr(in_tensor_dict["T_normal_B"])
        image_inter = np.concatenate(
            self.tensor2image(512, inter[0]) + [smpl_F, smpl_B, image], axis=1
        )

        Image.fromarray((image_inter).astype(np.uint8)).save(
            osp.join(self.export_dir, f"{mesh_rot}_inter.png")
        )

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_gt = batch["verts"][0]
        faces_gt = batch["faces"][0]

        self.result_eval.update(
            {
                "verts_gt": verts_gt,
                "faces_gt": faces_gt,
                "verts_pr": verts_pr,
                "faces_pr": faces_pr,
                "recon_size": (self.resolutions[-1] - 1.0),
                "calib": batch["calib"][0],
            }
        )

        self.evaluator.set_mesh(self.result_eval)
        # lap_op = self.evaluator.get_laplacian(self.evaluator.verts_gt, self.evaluator.faces_gt)
        chamfer, p2s = self.evaluator.calculate_chamfer_p2s(num_samples=1000)
        normal_consist = self.evaluator.calculate_normal_consist(
            osp.join(self.export_dir, f"{mesh_rot}_nc.png"))

        test_log = {"chamfer": chamfer, "p2s": p2s, "NC": normal_consist}

        self.log_dict(test_log, prog_bar=True, logger=False,
                      on_step=True, on_epoch=False)

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

        self.log_dict(accu_outputs, prog_bar=False,
                      logger=True, on_step=False, on_epoch=True)

        np.save(
            osp.join(self.export_dir, "../test_results.npy"),
            accu_outputs,
            allow_pickle=True,
        )

    def tensor2image(self, height, inter):

        all = []
        for dim in self.in_geo_dim:
            img = resize(
                np.tile(
                    ((inter[:dim].cpu().numpy() + 1.0) /
                     2.0 * 255.0).transpose(1, 2, 0),
                    (1, 1, int(3 / dim)),
                ),
                (height, height),
                anti_aliasing=True,
            )

            all.append(img)
            inter = inter[dim:]

        return all

    def render_func(self, in_tensor_dict, dataset="title", idx=0):

        for name in in_tensor_dict.keys():
            if in_tensor_dict[name] is not None:
                in_tensor_dict[name] = in_tensor_dict[name][0:1]

        self.netG.eval()
        features, inter = self.netG.filter(in_tensor_dict, return_inter=True)
        sdf = self.reconEngine(
            opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
        )

        if sdf is not None:
            render = self.reconEngine.display(sdf)

            image_pred = np.flip(render[:, :, ::-1], axis=0)
            height = image_pred.shape[0]

            image_gt = resize(
                ((in_tensor_dict["image"].cpu().numpy()[0] + 1.0) / 2.0 * 255.0).transpose(
                    1, 2, 0
                ),
                (height, height),
                anti_aliasing=True,
            )
            image_inter = self.tensor2image(height, inter[0])
            image = PIL.Image.fromarray(np.concatenate(
                [image_pred,
                 np.concatenate([image_gt]+image_inter+[image_gt], axis=1)],
                axis=0).astype(np.uint8))

            self.logger.experiment.log(
                {f"SDF/{dataset}/{idx}": wandb.Image(image, caption="multi-views")})

    def test_single(self, batch):

        self.netG.eval()
        self.netG.training = False
        in_tensor_dict = {}

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })
        
        with torch.no_grad():
            features, inter = self.netG.filter(
                in_tensor_dict, return_inter=True)
            sdf = self.reconEngine(
                opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
            )

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        return verts_pr, faces_pr, inter
