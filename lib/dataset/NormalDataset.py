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

import os.path as osp

import kornia
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from termcolor import colored


class NormalDataset:
    def __init__(self, cfg, split="train"):

        self.split = split
        self.root = cfg.root
        self.bsize = cfg.batch_size

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        self.scales = self.opt.scales

        # input data types and dimensions
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]
        self.in_total = self.in_nml + ["normal_F", "normal_B"]
        self.in_total_dim = self.in_nml_dim + [3, 3]

        if self.split != "train":
            self.rotations = range(0, 360, 120)
        else:
            self.rotations = np.arange(0, 360, 360 // self.opt.rotation_num).astype(np.int)

        self.datasets_dict = {}

        for dataset_id, dataset in enumerate(self.datasets):

            dataset_dir = osp.join(self.root, dataset)

            self.datasets_dict[dataset] = {
                "subjects": np.loadtxt(osp.join(dataset_dir, "all.txt"), dtype=str),
                "scale": self.scales[dataset_id],
            }

        self.subject_list = self.get_subject_list(split)

        # PIL to tensor
        self.image_to_tensor = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.0, ), (1.0, )),
            ]
        )

    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            split_txt = osp.join(self.root, dataset, f"{split}.txt")

            if osp.exists(split_txt) and osp.getsize(split_txt) > 0:
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        if self.split != "test":
            subject_list += subject_list[:self.bsize - len(subject_list) % self.bsize]
            print(colored(f"total: {len(subject_list)}", "yellow"))

        bug_list = sorted(np.loadtxt(osp.join(self.root, 'bug.txt'), dtype=str).tolist())

        subject_list = [subject for subject in subject_list if (subject not in bug_list)]

        # subject_list = ["thuman2/0008"]
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
            render_folder = "/".join([dataset + f"_36views", subject])

        # setup paths
        data_dict = {
            "dataset": dataset,
            "subject": subject,
            "rotation": rotation,
            "scale": self.datasets_dict[dataset]["scale"],
            "image_path": osp.join(self.root, render_folder, "render", f"{rotation:03d}.png"),
        }

        # image/normal/depth loader
        for name, channel in zip(self.in_total, self.in_total_dim):

            if f"{name}_path" not in data_dict.keys():
                data_dict.update(
                    {
                        f"{name}_path":
                            osp.join(self.root, render_folder, name, f"{rotation:03d}.png")
                    }
                )

            data_dict.update(
                {
                    name:
                        self.imagepath2tensor(
                            data_dict[f"{name}_path"], channel, inv=False, erasing=False
                        )
                }
            )

        path_keys = [key for key in data_dict.keys() if "_path" in key or "_dir" in key]

        for key in path_keys:
            del data_dict[key]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False, erasing=False):

        rgba = Image.open(path).convert("RGBA")
        mask = rgba.split()[-1]

        image = rgba.convert("RGB")
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)

        # simulate occlusion
        if erasing:
            mask = kornia.augmentation.RandomErasing(
                p=0.2, scale=(0.01, 0.2), ratio=(0.3, 3.3), keepdim=True
            )(mask)
        image = (image * mask)[:channel]

        return (image * (0.5 - inv) * 2.0).float()
