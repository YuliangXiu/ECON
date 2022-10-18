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

        super().__init__(cfg, split, vis)

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
                    texture_path = osp.join(self.datasets_dict[dataset]["mesh_dir"],
                                            f"{subject}/material0.jpeg")
                else:
                    mesh_path = osp.join(self.datasets_dict[dataset]["mesh_dir"], f"/{subject}.obj")
                    texture_path = osp.join(self.datasets_dict[dataset]["mesh_dir"],
                                            f"../raw_jpgs/{subject}.jpg")
                scale = self.scales[dataset_id]
                if subject not in self.mesh_cached[dataset].keys():
                    self.mesh_cached[dataset][subject] = self.load_textured_mesh(
                        mesh_path, scale, texture_path)

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

        # load training data
        data_dict.update(self.load_calib(data_dict))

        data_dict.update({
            "image": self.image2tensor(data_dict["image_path"], 'rgb'),
            "image_back": self.image2tensor(data_dict["image_back_path"], 'rgb'),
        })

        # data_dict.update(self.load_smpl(data_dict, self.vis))
        data_dict.update(self.get_sampling_color(data_dict))

        path_keys = [key for key in data_dict.keys() if "_path" in key or "_dir" in key]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    @staticmethod
    def load_textured_mesh(mesh_path, scale, texture_path):

        verts, faces, verts_uv, _ = obj_loader(mesh_path)
        texture = Image.open(texture_path)

        mesh = HoppeMesh(verts * scale, faces, verts_uv, texture)

        return mesh

    def get_sampling_color(self, data_dict):

        mesh = self.mesh_cached[data_dict['dataset']][data_dict['subject']]

        samples, face_index = sample_surface(mesh.triangles(), self.opt.num_sample_color)

        colors = mesh.get_colors(samples, mesh.faces[face_index])

        # Sampling offsets are random noise with constant scale (15cm - 20cm)
        offset = torch.normal(mean=0, std=self.opt.sigma_color, size=(self.opt.num_sample_color, 1))
        samples += mesh.face_normals[torch.tensor(face_index).long()] * offset.float()

        # Normalized to [-1, 1] rgb
        colors = ((colors[:, 0:3] / 255.0) - 0.5) / 0.5

        return {'samples_color': samples, 'labels_color': colors}

    def visualize_sampling3D(self, data_dict):

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg="white")
        vis_list = []

        points = projection(data_dict["samples_color"], data_dict["calib"])
        points[:, 1] *= -1
        colors = (data_dict["labels_color"] + 1.0) * 0.5
        
        # create a pointcloud
        pc = vedo.Points(points, r=15, c=np.float32(colors))
        vis_list.append(pc)

        # create a picure
        img_pos = [1.0, -1.0]
        for img_id, img_key in enumerate(["image", "image_back"]):
            image_arr = ((data_dict[img_key].repeat(3 // data_dict[img_key].shape[0], 1, 1).permute(
                1, 2, 0).numpy() + 1.0) * 0.5 * 255.0)
            image_dim = image_arr.shape[0]
            image = (vedo.Picture(image_arr).scale(2.0 / image_dim).pos(
                -1.0, -1.0, img_pos[img_id]))
            vis_list.append(image)

        vp.show(*vis_list, bg="white", axes=1.0, interactive=True)


if __name__ == "__main__":
    from lib.common.config import cfg
    from tqdm import tqdm
    cfg.merge_from_file("./configs/train/IF-Geo.yaml")
    cfg.freeze()
    dataset = TexDataset(cfg, "one", True)
    for data_dict in tqdm(dataset):
        # for key in data_dict.keys():
        #     if hasattr(data_dict[key], "shape"):
        #         print(key, data_dict[key].shape)

        dataset.visualize_sampling3D(data_dict)
        # break
