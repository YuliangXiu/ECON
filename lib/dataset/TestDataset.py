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

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pixielib.pixie import PIXIE
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.common.imutils import process_image
from lib.common.train_util import Format
from lib.net.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

from lib.pymafx.core import path_config
from lib.pymafx.models import pymaf_net

from lib.common.config import cfg
from lib.common.render import Render
from lib.dataset.body_model import TetraSMPLModel
from lib.dataset.mesh_util import get_visibility, SMPLX
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import detection

import os.path as osp
import torch
import glob
import numpy as np
from termcolor import colored
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset:
    def __init__(self, cfg, device):

        self.image_dir = cfg["image_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.use_seg = cfg["use_seg"]
        self.hps_type = cfg["hps_type"]
        self.smpl_type = "smplx"
        self.smpl_gender = "neutral"
        self.vol_res = cfg["vol_res"]
        self.single = cfg["single"]

        self.device = device

        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ["jpg", "png", "jpeg", "JPG", "bmp", "exr"]

        self.subject_list = sorted(
            [item for item in keep_lst if item.split(".")[-1] in img_fmts], reverse=False
        )

        # smpl related
        self.smpl_data = SMPLX()

        if self.hps_type == "pymafx":
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)["model"], strict=True)
            self.hps.eval()
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
        elif self.hps_type == "pixie":
            self.hps = PIXIE(config=pixie_cfg, device=self.device)

        self.smpl_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)

        self.detector = detection.maskrcnn_resnet50_fpn(
            weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights
        )
        self.detector.eval()

        print(
            colored(
                f"SMPL-X estimate with {Format.start} {self.hps_type.upper()} {Format.end}", "green"
            )
        )

        self.render = Render(size=512, device=self.device)

    def __len__(self):
        return len(self.subject_list)

    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=-1)
        smpl_vis = get_visibility(xy, z,
                                  torch.as_tensor(smpl_faces).long()[:, :,
                                                                     [0, 2, 1]]).unsqueeze(-1)
        smpl_cmap = self.smpl_data.cmap_smpl_vids(self.smpl_type).unsqueeze(0)

        return {
            "smpl_vis": smpl_vis.to(self.device),
            "smpl_cmap": smpl_cmap.to(self.device),
            "smpl_verts": smpl_verts,
        }

    def depth_to_voxel(self, data_dict):

        data_dict["depth_F"] = transforms.Resize(self.vol_res)(data_dict["depth_F"])
        data_dict["depth_B"] = transforms.Resize(self.vol_res)(data_dict["depth_B"])

        depth_mask = (~torch.isnan(data_dict['depth_F']))
        depth_FB = torch.cat([data_dict['depth_F'], data_dict['depth_B']], dim=0)
        depth_FB[:, ~depth_mask[0]] = 0.

        # Important: index_long = depth_value - 1
        index_z = (((depth_FB + 1.) * 0.5 * self.vol_res) - 1).clip(0, self.vol_res -
                                                                    1).permute(1, 2, 0)
        index_z_ceil = torch.ceil(index_z).long()
        index_z_floor = torch.floor(index_z).long()
        index_z_frac = torch.frac(index_z)

        index_mask = index_z[..., 0] == torch.tensor(self.vol_res * 0.5 - 1).long()
        voxels = F.one_hot(index_z_ceil[..., 0], self.vol_res) * index_z_frac[..., 0] + \
            F.one_hot(index_z_floor[..., 0], self.vol_res) * (1.0-index_z_frac[..., 0]) + \
            F.one_hot(index_z_ceil[..., 1], self.vol_res) * index_z_frac[..., 1]+ \
            F.one_hot(index_z_floor[..., 1], self.vol_res) * (1.0 - index_z_frac[..., 1])

        voxels[index_mask] *= 0
        voxels = torch.flip(voxels, [2]).permute(2, 0, 1).float()    #[x-2, y-0, z-1]

        return {
            "depth_voxels": voxels.flip([
                0,
            ]).unsqueeze(0).to(self.device),
        }

    def compute_voxel_verts(self, body_pose, global_orient, betas, trans, scale):

        smpl_path = osp.join(self.smpl_data.model_dir, "smpl/SMPL_NEUTRAL.pkl")
        tetra_path = osp.join(self.smpl_data.tedra_dir, "tetra_neutral_adult_smpl.npz")
        smpl_model = TetraSMPLModel(smpl_path, tetra_path, "adult")

        pose = torch.cat([global_orient[0], body_pose[0]], dim=0)
        smpl_model.set_params(rotation_matrix_to_angle_axis(rot6d_to_rotmat(pose)), beta=betas[0])

        verts = (
            np.concatenate([smpl_model.verts, smpl_model.verts_added], axis=0) * scale.item() +
            trans.detach().cpu().numpy()
        )
        faces = (
            np.loadtxt(
                osp.join(self.smpl_data.tedra_dir, "tetrahedrons_neutral_adult.txt"),
                dtype=np.int32,
            ) - 1
        )

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = (
            np.pad(verts, ((0, pad_v_num),
                           (0, 0)), mode="constant", constant_values=0.0).astype(np.float32) * 0.5
        )
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)), mode="constant",
                       constant_values=0.0).astype(np.int32)

        verts[:, 2] *= -1.0

        voxel_dict = {
            "voxel_verts": torch.from_numpy(verts).to(self.device).unsqueeze(0).float(),
            "voxel_faces": torch.from_numpy(faces).to(self.device).unsqueeze(0).long(),
            "pad_v_num": torch.tensor(pad_v_num).to(self.device).unsqueeze(0).long(),
            "pad_f_num": torch.tensor(pad_f_num).to(self.device).unsqueeze(0).long(),
        }

        return voxel_dict

    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        arr_dict = process_image(img_path, self.hps_type, self.single, 512, self.detector)
        arr_dict.update({"name": img_name})

        with torch.no_grad():
            if self.hps_type == "pixie":
                preds_dict = self.hps.forward(arr_dict["img_hps"].to(self.device))
            elif self.hps_type == 'pymafx':
                batch = {k: v.to(self.device) for k, v in arr_dict["img_pymafx"].items()}
                preds_dict, _ = self.hps.forward(batch)

        arr_dict["smpl_faces"] = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(
                self.device
            )
        )
        arr_dict["type"] = self.smpl_type

        if self.hps_type == "pymafx":
            output = preds_dict["mesh_out"][-1]
            scale, tranX, tranY = output["theta"][:, :3].split(1, dim=1)
            arr_dict["betas"] = output["pred_shape"]
            arr_dict["body_pose"] = output["rotmat"][:, 1:22]
            arr_dict["global_orient"] = output["rotmat"][:, 0:1]
            arr_dict["smpl_verts"] = output["smplx_verts"]
            arr_dict["left_hand_pose"] = output["pred_lhand_rotmat"]
            arr_dict["right_hand_pose"] = output["pred_rhand_rotmat"]
            arr_dict['jaw_pose'] = output['pred_face_rotmat'][:, 0:1]
            arr_dict["exp"] = output["pred_exp"]
            # 1.2009, 0.0013, 0.3954

        elif self.hps_type == "pixie":
            arr_dict.update(preds_dict)
            arr_dict["global_orient"] = preds_dict["global_pose"]
            arr_dict["betas"] = preds_dict["shape"]    #200
            arr_dict["smpl_verts"] = preds_dict["vertices"]
            scale, tranX, tranY = preds_dict["cam"].split(1, dim=1)
            # 1.1435, 0.0128, 0.3520

        arr_dict["scale"] = scale.unsqueeze(1)
        arr_dict["trans"] = (
            torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                      dim=1).unsqueeze(1).to(self.device).float()
        )

        # data_dict info (key-shape):
        # scale, tranX, tranY - tensor.float
        # betas - [1,10] / [1, 200]
        # body_pose - [1, 23, 3, 3] / [1, 21, 3, 3]
        # global_orient - [1, 1, 3, 3]
        # smpl_verts - [1, 6890, 3] / [1, 10475, 3]

        # from rot_mat to rot_6d for better optimization
        N_body, N_pose = arr_dict["body_pose"].shape[:2]
        arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
        arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

        return arr_dict

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")
