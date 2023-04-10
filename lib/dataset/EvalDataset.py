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

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import trimesh
from PIL import Image

from lib.common.render import Render
from lib.dataset.mesh_util import SMPLX, HoppeMesh, projection, rescale_smpl

cape_gender = {
    "male":
    ['00032', '00096', '00122', '00127', '00145', '00215', '02474', '03284', '03375',
     '03394'], "female": ['00134', '00159', '03223', '03331', '03383']
}


class EvalDataset:
    def __init__(self, cfg, device):

        self.root = cfg.root
        self.bsize = cfg.batch_size

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        self.scales = self.opt.scales
        self.vol_res = cfg.vol_res

        # [(feat_name, channel_num),...]
        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]

        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]

        self.in_total = self.in_geo + self.in_nml
        self.in_total_dim = self.in_geo_dim + self.in_nml_dim

        self.rotations = range(0, 360, 120)

        self.datasets_dict = {}

        for dataset_id, dataset in enumerate(self.datasets):

            dataset_dir = osp.join(self.root, dataset)

            mesh_dir = osp.join(dataset_dir, "scans")
            smplx_dir = osp.join(dataset_dir, "smplx")
            smpl_dir = osp.join(dataset_dir, "smpl")

            self.datasets_dict[dataset] = {
                "smplx_dir": smplx_dir,
                "smpl_dir": smpl_dir,
                "mesh_dir": mesh_dir,
                "scale": self.scales[dataset_id],
            }

            self.datasets_dict[dataset].update({
                "subjects":
                np.loadtxt(osp.join(dataset_dir, "all.txt"), dtype=str)
            })

        self.subject_list = self.get_subject_list()
        self.smplx = SMPLX()

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0, ), (1.0, )),
        ])

        self.device = device
        self.render = Render(size=512, device=self.device)

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image()

    def get_subject_list(self):

        subject_list = []

        for dataset in self.datasets:

            split_txt = ""

            if dataset == 'renderpeople':
                split_txt = osp.join(self.root, dataset, "loose.txt")
            elif dataset == 'cape':
                split_txt = osp.join(self.root, dataset, "pose.txt")

            if osp.exists(split_txt) and osp.getsize(split_txt) > 0:
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations)

    def __getitem__(self, index):

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset = self.subject_list[mid].split("/")[0]
        render_folder = "/".join([dataset + f"_{self.opt.rotation_num}views", subject])

        if not osp.exists(osp.join(self.root, render_folder)):
            render_folder = "/".join([dataset + "_36views", subject])

        # setup paths
        data_dict = {
            "dataset": dataset,
            "subject": subject,
            "rotation": rotation,
            "scale": self.datasets_dict[dataset]["scale"],
            "calib_path": osp.join(self.root, render_folder, "calib", f"{rotation:03d}.txt"),
            "image_path": osp.join(self.root, render_folder, "render", f"{rotation:03d}.png"),
        }

        if dataset == "cape":
            data_dict.update({
                "mesh_path":
                osp.join(self.datasets_dict[dataset]["mesh_dir"], f"{subject}.obj"),
                "smpl_path":
                osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.obj"),
            })
        else:

            data_dict.update({
                "mesh_path":
                osp.join(
                    self.datasets_dict[dataset]["mesh_dir"],
                    f"{subject}.obj",
                ),
                "smplx_path":
                osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}.obj"),
            })

        # load training data
        data_dict.update(self.load_calib(data_dict))

        # image/normal/depth loader
        for name, channel in zip(self.in_total, self.in_total_dim):

            if f"{name}_path" not in data_dict.keys():
                data_dict.update({
                    f"{name}_path":
                    osp.join(self.root, render_folder, name, f"{rotation:03d}.png")
                })

            # tensor update
            if os.path.exists(data_dict[f"{name}_path"]):
                data_dict.update({
                    name:
                    self.imagepath2tensor(data_dict[f"{name}_path"], channel, inv=False)
                })

        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(self.load_smpl(data_dict))

        del data_dict["mesh"]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False):

        rgba = Image.open(path).convert("RGBA")

        # remove CAPE's noisy outliers using OpenCV's inpainting
        if "cape" in path and "T_" not in path:
            mask = cv2.imread(path.replace(path.split("/")[-2], "mask"), 0) > 1
            img = np.asarray(rgba)[:, :, :3]
            fill_mask = ((mask & (img.sum(axis=2) == 0))).astype(np.uint8)
            image = Image.fromarray(
                cv2.inpaint(img * mask[..., None], fill_mask, 3, cv2.INPAINT_TELEA)
            )
            mask = Image.fromarray(mask)
        else:
            mask = rgba.split()[-1]
            image = rgba.convert("RGB")

        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]

        return (image * (0.5 - inv) * 2.0).float()

    def load_calib(self, data_dict):
        calib_data = np.loadtxt(data_dict["calib_path"], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {"calib": calib_mat}

    def load_mesh(self, data_dict):

        mesh_path = data_dict["mesh_path"]
        scale = data_dict["scale"]

        scan_mesh = trimesh.load(mesh_path)
        verts = scan_mesh.vertices
        faces = scan_mesh.faces

        mesh = HoppeMesh(verts * scale, faces)

        return {
            "mesh": mesh,
            "verts": torch.as_tensor(verts * scale).float(),
            "faces": torch.as_tensor(faces).long(),
        }

    def load_smpl(self, data_dict):

        smpl_type = ("smplx" if ("smplx_path" in data_dict.keys()) else "smpl")

        smplx_verts = rescale_smpl(data_dict[f"{smpl_type}_path"], scale=100.0)
        smplx_faces = torch.as_tensor(getattr(self.smplx, f"{smpl_type}_faces")).long()
        smplx_verts = projection(smplx_verts, data_dict["calib"]).float()

        return_dict = {
            "smpl_verts": smplx_verts,
            "smpl_faces": smplx_faces,
        }

        return return_dict

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

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")
