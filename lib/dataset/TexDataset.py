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

from lib.renderer.mesh import load_fit_body
from lib.dataset.mesh_util import *
from lib.dataset.IFDataset import IFDataset
from termcolor import colored
import os.path as osp
import numpy as np
from PIL import Image
import os
import kornia
import trimesh
import torch
import torch.nn.functional as F
import vedo
from tqdm import tqdm
import torchvision.transforms as transforms


class TexDataset(IFDataset):

    def __init__(self, cfg, split="train", vis=False):
        
        super(IFDataset, self).__init__(cfg, split="train", vis=False)

        self.mesh_cached = {}

        for dataset_id, dataset in enumerate(self.datasets):

            # pre-cached meshes
            self.mesh_cached[dataset] = {}
            pbar = tqdm(self.datasets_dict[dataset]["subjects"])
            for subject in pbar:
                subject = subject.split("/")[-1]
                pbar.set_description(f"Loading {dataset}-{split}-{subject}")
                if dataset == "thuman2":
                    mesh_path = osp.join(self.datasets_dict[dataset]["mesh_dir"],
                                         f"{subject}/{subject}.obj")
                else:
                    mesh_path = osp.join(self.datasets_dict[dataset]["mesh_dir"], f"/{subject}.obj")
                scale = self.scales[dataset_id]
                if subject not in self.mesh_cached[dataset].keys():
                    self.mesh_cached[dataset][subject] = self.load_textured_mesh(mesh_path, scale)


    def __getitem__(self, index):

        # only pick the first data if overfitting
        if self.overfit:
            index = 0

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset = self.subject_list[mid].split("/")[0]
        render_folder = "/".join([dataset + f"_{self.opt.rotation_num}views", subject])

        # setup paths
        data_dict = {
            "dataset":
                dataset,
            "subject":
                subject,
            "rotation":
                rotation,
            "scale":
                self.datasets_dict[dataset]["scale"],
            "calib_path":
                osp.join(self.root, render_folder, "calib", f"{rotation:03d}.txt"),
            "image_path":
                osp.join(self.root, render_folder, "render", f"{rotation:03d}.png"),
            "image_back_path":
                osp.join(self.root, render_folder, "render", f"{(180-rotation)%360:03d}.png"),
        }

        if dataset == "thuman2":
            data_dict.update({
                "smplx_path":
                    osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}.obj"),
                "smplx_param":
                    osp.join(
                        self.datasets_dict[dataset]["smplx_dir"],
                        f"{subject}.pkl",
                    ),
                "joint_path":
                    osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}_joints.npy"),
                "smpl_path":
                    osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.obj"),
                "smpl_param":
                    osp.join(
                        self.datasets_dict[dataset]["smpl_dir"],
                        f"{subject}.pkl",
                    ),
            })

        elif dataset == "cape":
            data_dict.update({
                "image_back_path":
                    osp.join(self.root, render_folder, "render", f"{rotation:03d}.png"),
                "joint_path":
                    osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.npy"),
                "smpl_path":
                    osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.obj"),
                "smpl_param":
                    osp.join(
                        self.datasets_dict[dataset]["smpl_dir"],
                        f"{subject}.npz",
                    ),
            })
        else:

            data_dict.update({
                "smplx_path":
                    osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}.obj"),
                "smplx_param":
                    osp.join(
                        self.datasets_dict[dataset]["smplx_dir"],
                        f"{subject}.pkl",
                    ),
                "joint_path":
                    osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}_joints.npy"),
                "smpl_path":
                    osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.obj"),
                "smpl_param":
                    osp.join(
                        self.datasets_dict[dataset]["smpl_dir"],
                        f"{subject}.pkl",
                    ),
            })

        # load training data
        data_dict.update(self.load_calib(data_dict))

        data_dict.update({
            "image": self.image2tensor(data_dict["image_path"], 'rgb'),
            "image_back": self.image2tensor(data_dict["image_back_path"], 'rgb'),
            "depth_F": self.image2tensor(data_dict["image_path"].replace("render", "depth_F"), 'z'),
            "depth_B": self.image2tensor(data_dict["image_path"].replace("render", "depth_B"), 'z')
        })

        # data_dict.update(self.load_smpl(data_dict, self.vis))
        data_dict.update(self.get_sampling_geo(data_dict))

        # load voxels from full SMPL body
        data_dict.update(self.compute_voxel_verts(data_dict))

        # load voxels from partial depth
        data_dict.update(self.depth_to_voxel(data_dict))

        path_keys = [key for key in data_dict.keys() if "_path" in key or "_dir" in key]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    def image2tensor(self, path, type='rgb'):

        if type == 'rgb':
            rgba = Image.open(path).convert("RGBA")
            mask = rgba.split()[-1]
            image = rgba.convert("RGB")
            image = self.transform_to_tensor(res=self.img_res,
                                             mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))(image)
            mask = self.transform_to_tensor(res=self.img_res, mean=(0.0,), std=(1.0,))(mask)
            out = (image * mask)[:3].float()
        else:
            _, _, depth, mask = Image.open(path).split()
            depth = self.transform_to_tensor(res=self.vol_res, mean=0.5, std=0.5)(depth)
            mask = self.transform_to_tensor(res=self.vol_res, mean=(0.0,), std=(1.0,))(mask)
            out = (depth * (mask == 1.)).float()

        return out


    @staticmethod
    def load_textured_mesh(mesh_path, scale):

        verts, faces = obj_loader(mesh_path)

        mesh = HoppeMesh(verts * scale, faces)

        return mesh

    def get_sampling_tex(self, data_dict):

        mesh = self.mesh_cached[data_dict['dataset']][data_dict['subject']]
        calib = data_dict["calib"]

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        vert_ids = np.arange(mesh.verts.shape[0])

        samples_surface_ids = np.random.choice(vert_ids, n_samples_surface, replace=True)

        samples_surface = mesh.verts[samples_surface_ids, :]

        # Sampling offsets are random noise with constant scale (15cm - 20cm)
        offset = np.random.normal(scale=self.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.vert_normals[samples_surface_ids, :] * offset

        # Uniform samples in [-1, 1]
        calib_inv = np.linalg.inv(calib)
        n_samples_space = self.opt.num_sample_geo // 4
        samples_space_img = 2.0 * np.random.rand(n_samples_space, 3) - 1.0
        samples_space = projection(samples_space_img, calib_inv)

        samples = np.concatenate([samples_surface, samples_space], 0)
        np.random.shuffle(samples)

        # labels: in->1.0; out->0.0.
        inside = mesh.contains(samples)
        inside_samples = samples[inside >= 0.5]
        outside_samples = samples[inside < 0.5]

        nin = inside_samples.shape[0]

        if nin > self.opt.num_sample_geo // 2:
            inside_samples = inside_samples[:self.opt.num_sample_geo // 2]
            outside_samples = outside_samples[:self.opt.num_sample_geo // 2]
        else:
            outside_samples = outside_samples[:(self.opt.num_sample_geo - nin)]

        samples = np.concatenate([inside_samples, outside_samples])
        labels = np.concatenate(
            [np.ones(inside_samples.shape[0]),
             np.zeros(outside_samples.shape[0])])

        samples = torch.from_numpy(samples).float()
        labels = torch.from_numpy(labels).float()

        return {"samples_geo": samples, "labels_geo": labels}

    def visualize_sampling3D(self, data_dict, mode="vis"):

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg="white")
        vis_list = []

        assert mode in ["occ", "kpt"]

        if mode == "occ":
            labels = data_dict[f"labels_geo"][..., None]  # occupancy
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == "kpt":
            colors = np.ones_like(data_dict["smpl_joint"])

        points = projection(data_dict["samples_geo"], data_dict["calib"])
        verts = projection(self.mesh_cached[data_dict['dataset']][data_dict['subject']]["verts"],
                           data_dict["calib"])
        points[:, 1] *= -1
        verts[:, 1] *= -1

        keypoints = data_dict["smpl_joint"]
        keypoints[:, 1] *= -1  # [-1,1] 24x3

        # create a mesh
        if mode == 'kpt':
            alpha = 0.2
        else:
            alpha = 1.0

        mesh = trimesh.Trimesh(
            verts,
            self.mesh_cached[data_dict['dataset']][data_dict['subject']]["faces"],
            process=True)
        mesh.visual.vertex_colors = [128.0, 128.0, 128.0, alpha * 255.0]
        vis_list.append(mesh)
        
        if mode != 'kpt':
            # create a pointcloud
            pc = vedo.Points(points, r=15, c=np.float32(colors))
            vis_list.append(pc)

        # create a picure
        img_pos = [1.0, 0.0, -1.0]
        for img_id, img_key in enumerate(["depth_F", "image", "depth_B"]):
            image_arr = ((data_dict[img_key].repeat(3 // data_dict[img_key].shape[0], 1, 1).permute(
                1, 2, 0).numpy() + 1.0) * 0.5 * 255.0)
            image_dim = image_arr.shape[0]
            image = (vedo.Picture(image_arr).scale(2.0 / image_dim).pos(
                -1.0, -1.0, img_pos[img_id]))
            vis_list.append(image)

        # create skeleton
        kpt = vedo.Points(keypoints, r=15, c=np.float32(np.ones_like(keypoints)))
        vis_list.append(kpt)

        vp.show(*vis_list, bg="white", axes=1.0, interactive=True)


if __name__ == "__main__":
    from lib.common.config import cfg
    from tqdm import tqdm
    cfg.merge_from_file("./configs/train/IF-Geo.yaml")
    cfg.freeze()
    dataset = IFDataset(cfg, "train", False)
    for i, data_dict in enumerate(tqdm(dataset[:3])):
        pass
        # for key in data_dict.keys():
        #     if hasattr(data_dict[key], "shape"):
        #         print(key, data_dict[key].shape)

        # calib torch.Size([4, 4])
        # image torch.Size([3, 512, 512])
        # depth_F torch.Size([1, 128, 128])
        # depth_B torch.Size([1, 128, 128])
        # verts torch.Size([302021, 3])
        # faces torch.Size([498850, 3])
        # smpl_joint torch.Size([24, 3])
        # smpl_verts torch.Size([10475, 3])
        # smpl_faces torch.Size([20908, 3])
        # samples_geo torch.Size([8000, 3])
        # labels_geo torch.Size([8000])
        # voxel_verts torch.Size([8000, 3])
        # voxel_faces (25100, 4)
        # depth_voxels torch.Size([128, 128, 128])

        # dataset.visualize_sampling3D(data_dict, mode="occ")
        # break
