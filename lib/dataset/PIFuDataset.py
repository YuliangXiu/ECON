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

from lib.renderer.mesh import load_fit_body, compute_normal_batch
from lib.dataset.body_model import TetraSMPLModel
from lib.common.render import Render
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import (
    SMPLX,
    projection,
    get_visibility,
    rescale_smpl,
    HoppeMesh,
    obj_loader,
)
from lib.net.geometry import rotation_matrix_to_angle_axis
from termcolor import colored
import os.path as osp
import numpy as np
from PIL import Image
import os
import cv2
import trimesh
import torch
import vedo
import torchvision.transforms as transforms

cape_gender = {
    "male": [
        '00032', '00096', '00122', '00127', '00145', '00215', '02474', '03284',
        '03375', '03394'
    ],
    "female": ['00134', '00159', '03223', '03331', '03383']
}


class PIFuDataset:

    def __init__(self, cfg, split="train", vis=False):

        self.split = split
        self.root = cfg.root
        self.bsize = cfg.batch_size
        self.overfit = cfg.overfit
        self.laplacian_iters = cfg.dataset.laplacian_iters

        # for debug, only used in visualize_sampling3D
        self.vis = vis

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        self.scales = self.opt.scales
        self.workers = cfg.num_threads
        self.prior_type = cfg.net.prior_type

        self.base_keys = ["smpl_verts", "smpl_faces"]
        self.icon_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in cfg.net.smpl_feats
        ]
        self.keypoint_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in cfg.net.smpl_feats
        ]
        self.pamir_keys = [
            "voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"
        ]
        self.pifu_keys = []

        self.noise_type = self.opt.noise_type
        self.noise_scale = self.opt.noise_scale

        noise_joints = [4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 21]
        self.smpl_joint_ids = np.arange(22).tolist() + [68, 73]

        self.noise_smpl_idx = []
        self.noise_smplx_idx = []

        for idx in noise_joints:
            self.noise_smpl_idx.append(idx * 3)
            self.noise_smpl_idx.append(idx * 3 + 1)
            self.noise_smpl_idx.append(idx * 3 + 2)

            self.noise_smplx_idx.append((idx - 1) * 3)
            self.noise_smplx_idx.append((idx - 1) * 3 + 1)
            self.noise_smplx_idx.append((idx - 1) * 3 + 2)

        self.use_sdf = cfg.sdf
        self.sdf_clip = cfg.sdf_clip

        # [(feat_name, channel_num),...]
        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]

        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]

        self.in_total = self.in_geo + self.in_nml
        self.in_total_dim = self.in_geo_dim + self.in_nml_dim

        if self.split == "train":
            self.rotations = np.arange(0, 360, 360 /
                                       self.opt.rotation_num).astype(np.int32)
        else:
            self.rotations = range(0, 360, 120)

        self.datasets_dict = {}

        for dataset_id, dataset in enumerate(self.datasets):

            mesh_dir = None
            smplx_dir = None

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

        self.subject_list = self.get_subject_list(split)
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

        self.device = torch.device(f"cuda:{cfg.gpus[0]}")
        self.render = Render(size=512, device=self.device)

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image()

    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            split_txt = osp.join(self.root, dataset, f"{split}.txt")

            if osp.exists(split_txt) and osp.getsize(split_txt) > 0:
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        if self.split == "train":
            subject_list += subject_list[:self.bsize -
                                         len(subject_list) % self.bsize]
            print(colored(f"total: {len(subject_list)}", "yellow"))

        # subject_list = ["thuman2/0008"]
        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations)

    def __getitem__(self, index):

        # only pick the first data if overfitting
        if self.overfit:
            index = 0

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset = self.subject_list[mid].split("/")[0]
        render_folder = "/".join(
            [dataset + f"_{self.opt.rotation_num}views", subject])
        

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
            osp.join(self.root, render_folder, "render",
                     f"{rotation:03d}.png"),
            "vis_path":
            osp.join(self.root, render_folder, "vis", f"{rotation:03d}.pt"),
        }

        if dataset == "thuman2":
            data_dict.update({
                "mesh_path":
                osp.join(
                    self.datasets_dict[dataset]["mesh_dir"],
                    f"{subject}/{subject}.obj",
                ),
                "smplx_path":
                osp.join(self.datasets_dict[dataset]["smplx_dir"],
                         f"{subject}.obj"),
                "smplx_param":
                osp.join(
                    self.datasets_dict[dataset]["smplx_dir"],
                    f"{subject}.pkl",
                ),
                "joint_path":
                osp.join(self.datasets_dict[dataset]["smplx_dir"],
                         f"{subject}_joints.npy"),
                "smpl_path":
                osp.join(self.datasets_dict[dataset]["smpl_dir"],
                         f"{subject}.obj"),
                "smpl_param":
                osp.join(
                    self.datasets_dict[dataset]["smpl_dir"],
                    f"{subject}.pkl",
                ),
            })

        elif dataset == "cape":
            data_dict.update({
                "mesh_path":
                osp.join(self.datasets_dict[dataset]["mesh_dir"],
                         f"{subject}.obj"),
                "joint_path":
                osp.join(self.datasets_dict[dataset]["smpl_dir"],
                         f"{subject}.npy"),
                "smpl_path":
                osp.join(self.datasets_dict[dataset]["smpl_dir"],
                         f"{subject}.obj"),
                "smpl_param":
                osp.join(
                    self.datasets_dict[dataset]["smpl_dir"],
                    f"{subject}.npz",
                ),
            })
        else:

            data_dict.update({
                "mesh_path":
                osp.join(
                    self.datasets_dict[dataset]["mesh_dir"],
                    f"{subject}.obj",
                ),
                "smplx_path":
                osp.join(self.datasets_dict[dataset]["smplx_dir"],
                         f"{subject}.obj"),
                "smplx_param":
                osp.join(
                    self.datasets_dict[dataset]["smplx_dir"],
                    f"{subject}.pkl",
                ),
                "joint_path":
                osp.join(self.datasets_dict[dataset]["smplx_dir"],
                         f"{subject}_joints.npy"),
                "smpl_path":
                osp.join(self.datasets_dict[dataset]["smpl_dir"],
                         f"{subject}.obj"),
                "smpl_param":
                osp.join(
                    self.datasets_dict[dataset]["smpl_dir"],
                    f"{subject}.pkl",
                ),
            })

        # load training data
        data_dict.update(self.load_calib(data_dict))

        # image/normal/depth loader
        for name, channel in zip(self.in_total, self.in_total_dim):

            if f"{name}_path" not in data_dict.keys():
                data_dict.update({
                    f"{name}_path":
                    osp.join(self.root, render_folder, name,
                             f"{rotation:03d}.png")
                })

            # tensor update
            if os.path.exists(data_dict[f"{name}_path"]):
                data_dict.update({
                    name:
                    self.imagepath2tensor(data_dict[f"{name}_path"],
                                          channel,
                                          inv=False)
                })

        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(
            self.get_sampling_geo(data_dict,
                                  is_valid=self.split == "val",
                                  is_sdf=self.use_sdf))
        data_dict.update(self.load_smpl(data_dict, self.vis))

        if self.prior_type == "pamir":
            data_dict.update(self.load_smpl_voxel(data_dict))

        if (self.split != "test") and (not self.vis):

            del data_dict["verts"]
            del data_dict["faces"]

        if not self.vis:
            del data_dict["mesh"]

        path_keys = [
            key for key in data_dict.keys() if "_path" in key or "_dir" in key
        ]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False):

        rgba = Image.open(path).convert("RGBA")

        # remove CAPE's noisy outliers using OpenCV's inpainting
        if "cape" in path and "T_" not in path:
            mask = cv2.imread(path.replace(path.split("/")[-2], "mask"), 0) > 1
            img = np.asarray(rgba)[:, :, :3]
            fill_mask = ((mask & (img.sum(axis=2) == 0))).astype(np.uint8)
            image = Image.fromarray(
                cv2.inpaint(img * mask[..., None], fill_mask, 3,
                            cv2.INPAINT_TELEA))
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

        verts, faces = obj_loader(mesh_path, with_uv=False)

        mesh = HoppeMesh(verts * scale, faces)

        return {
            "mesh": mesh,
            "verts": torch.as_tensor(verts * scale).float(),
            "faces": torch.as_tensor(faces).long(),
        }

    def add_noise(self, beta_num, smpl_pose, smpl_betas, noise_type,
                  noise_scale, type, hashcode):

        # np.random.seed(hashcode)

        if type == "smplx":
            noise_idx = self.noise_smplx_idx
        else:
            noise_idx = self.noise_smpl_idx

        if "beta" in noise_type and noise_scale[noise_type.index(
                "beta")] > 0.0:
            smpl_betas += ((np.random.rand(beta_num) - 0.5) * 2.0 *
                           noise_scale[noise_type.index("beta")])
            smpl_betas = smpl_betas.astype(np.float32)

        if "pose" in noise_type and noise_scale[noise_type.index(
                "pose")] > 0.0:
            smpl_pose[noise_idx] += ((np.random.rand(len(noise_idx)) - 0.5) *
                                     2.0 * np.pi *
                                     noise_scale[noise_type.index("pose")])
            smpl_pose = smpl_pose.astype(np.float32)
        if type == "smplx":
            return torch.as_tensor(smpl_pose[None, ...]), torch.as_tensor(
                smpl_betas[None, ...])
        else:
            return smpl_pose, smpl_betas

    def compute_smpl_verts(self, data_dict, noise_type=None, noise_scale=None):

        dataset = data_dict["dataset"]
        smplx_dict = {}

        smplx_param = np.load(data_dict["smplx_param"], allow_pickle=True)
        smplx_pose = smplx_param["body_pose"]  # [1,63]
        smplx_betas = smplx_param["betas"]  # [1,10]
        smplx_pose, smplx_betas = self.add_noise(
            smplx_betas.shape[1],
            smplx_pose[0],
            smplx_betas[0],
            noise_type,
            noise_scale,
            type="smplx",
            hashcode=(hash(f"{data_dict['subject']}_{data_dict['rotation']}"))
            % (10**8),
        )

        smplx_out, smplx_joints = load_fit_body(
            fitted_path=data_dict["smplx_param"],
            scale=self.datasets_dict[dataset]["scale"],
            smpl_type="smplx",
            smpl_gender="male",
            noise_dict=dict(betas=smplx_betas, body_pose=smplx_pose),
        )

        smplx_dict.update({
            "type": "smplx",
            "gender": "male",
            "body_pose": torch.as_tensor(smplx_pose),
            "betas": torch.as_tensor(smplx_betas),
            "smpl_joint": torch.as_tensor(smplx_joints).float(),
        })

        return smplx_out.vertices, smplx_dict

    def compute_voxel_verts(self,
                            data_dict,
                            noise_type=None,
                            noise_scale=None):

        smpl_param = np.load(data_dict["smpl_param"], allow_pickle=True)

        if data_dict['dataset'] == 'cape':
            pid = data_dict['subject'].split("-")[0]
            gender = "male" if pid in cape_gender["male"] else "female"
            smpl_pose = smpl_param['pose'].flatten()
            smpl_betas = np.zeros((1, 10))
        else:
            gender = 'male'
            smpl_pose = rotation_matrix_to_angle_axis(
                torch.as_tensor(smpl_param["full_pose"][0])).numpy()
            smpl_betas = smpl_param["betas"]

        smpl_path = osp.join(self.smplx.model_dir,
                             f"smpl/SMPL_{gender.upper()}.pkl")
        tetra_path = osp.join(self.smplx.tedra_dir,
                              f"tetra_{gender}_adult_smpl.npz")

        smpl_model = TetraSMPLModel(smpl_path, tetra_path, "adult")

        smpl_pose, smpl_betas = self.add_noise(
            smpl_model.beta_shape[0],
            smpl_pose.flatten(),
            smpl_betas[0],
            noise_type,
            noise_scale,
            type="smpl",
            hashcode=(hash(f"{data_dict['subject']}_{data_dict['rotation']}"))
            % (10**8),
        )

        smpl_model.set_params(pose=smpl_pose.reshape(-1, 3),
                              beta=smpl_betas,
                              trans=smpl_param["transl"])
        if data_dict['dataset'] == 'cape':
            verts = np.concatenate([smpl_model.verts, smpl_model.verts_added],
                                   axis=0) * 100.0
        else:
            verts = (np.concatenate([smpl_model.verts, smpl_model.verts_added],
                                    axis=0) * smpl_param["scale"] +
                     smpl_param["translation"]
                     ) * self.datasets_dict[data_dict["dataset"]]["scale"]

        faces = (np.loadtxt(
            osp.join(self.smplx.tedra_dir, "tetrahedrons_male_adult.txt"),
            dtype=np.int32,
        ) - 1)

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = np.pad(verts, ((0, pad_v_num), (0, 0)),
                       mode="constant",
                       constant_values=0.0).astype(np.float32)
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)),
                       mode="constant",
                       constant_values=0.0).astype(np.int32)

        return verts, faces, pad_v_num, pad_f_num

    def load_smpl(self, data_dict, vis=False):

        smpl_type = ("smplx" if
                     ("smplx_path" in data_dict.keys()
                      and os.path.exists(data_dict["smplx_path"])) else "smpl")

        return_dict = {}

        # add random noise to SMPL-(X) params
        if ("smplx_param" in data_dict.keys()
                and os.path.exists(data_dict["smplx_param"])
                and sum(self.noise_scale) > 0.0):
            smplx_verts, smplx_dict = self.compute_smpl_verts(
                data_dict, self.noise_type, self.noise_scale)
            smplx_faces = torch.as_tensor(self.smplx.smplx_faces).long()

            if "smpl_vis" in getattr(self, f"{self.prior_type}_keys"):

                (xy, z) = (torch.as_tensor(smplx_verts).to(self.device).split(
                    [2, 1], dim=1))
                smplx_vis = get_visibility(
                    xy, z,
                    torch.as_tensor(smplx_faces).to(self.device).long())

                return_dict.update({"smpl_vis": smplx_vis})

            return_dict.update(smplx_dict)

        # instead, directly load SMPL-(X) objs
        else:
            smplx_vis = torch.load(data_dict["vis_path"]).float()
            smplx_joints = (
                torch.as_tensor(np.load(data_dict["joint_path"])).float() *
                100.0)
            return_dict.update({
                "smpl_vis": smplx_vis,
                "smpl_joint": smplx_joints
            })

            smplx_verts = rescale_smpl(data_dict[f"{smpl_type}_path"],
                                       scale=100.0)
            smplx_faces = torch.as_tensor(
                getattr(self.smplx, f"{smpl_type}_faces")).long()

        # skeleton loading
        smplx_joints = projection(return_dict["smpl_joint"],
                                  data_dict["calib"]).float()

        if smpl_type == "smplx":
            return_dict.update(
                {"smpl_joint": smplx_joints[self.smpl_joint_ids, :]})
        else:
            return_dict.update({"smpl_joint": smplx_joints[:24, :]})

        smplx_cmap = self.smplx.cmap_smpl_vids(smpl_type)
        smplx_verts = projection(smplx_verts, data_dict["calib"]).float()
        smplx_norm = torch.as_tensor(
            compute_normal_batch(smplx_verts[None, ...],
                                 smplx_faces[None, ...])).float()[0]

        if "smpl_cmap" in getattr(self, f"{self.prior_type}_keys"):
            return_dict.update({"smpl_cmap": smplx_cmap})
        if "smpl_norm" in getattr(self, f"{self.prior_type}_keys"):
            return_dict.update({"smpl_norm": smplx_norm})

        return_dict.update({
            "smpl_verts": smplx_verts,
            "smpl_faces": smplx_faces,
        })

        if vis:

            T_normal_F, T_normal_B = self.render_normal(
                (smplx_verts * torch.tensor(np.array([1.0, -1.0, 1.0]))).to(
                    self.device),
                smplx_faces.to(self.device),
            )

            return_dict.update({
                "T_normal_F": T_normal_F.squeeze(0),
                "T_normal_B": T_normal_B.squeeze(0),
            })

            query_points = projection(data_dict["samples_geo"],
                                      data_dict["calib"]).float()

            point_feat_extractor = PointFeat(
                smplx_verts.unsqueeze(0).to(self.device),
                smplx_faces.unsqueeze(0).to(self.device),
            )

            point_feat_out = point_feat_extractor.query(
                query_points.unsqueeze(0).contiguous().to(self.device),
                {
                    "smpl_sdf": None,
                    "smpl_cmap": smplx_cmap.unsqueeze(0).to(self.device),
                    "smpl_norm": smplx_norm.unsqueeze(0).to(self.device),
                    "smpl_vis": smplx_vis.unsqueeze(0).to(self.device),
                },
            )

            feat_lst = [
                point_feat_out[key][0].detach().cpu()
                for key in ["sdf", "cmap", "norm", "vis"]
            ]

            return_dict.update({"smpl_feat": torch.cat(feat_lst, dim=1)})

        return return_dict

    def load_smpl_voxel(self, data_dict):

        smpl_verts, smpl_faces, pad_v_num, pad_f_num = self.compute_voxel_verts(
            data_dict, self.noise_type,
            self.noise_scale)  # compute using smpl model
        smpl_verts = projection(smpl_verts, data_dict["calib"])

        smpl_verts *= 0.5

        return {
            "voxel_verts": smpl_verts,
            "voxel_faces": smpl_faces,
            "pad_v_num": pad_v_num,
            "pad_f_num": pad_f_num,
        }

    def get_sampling_geo(self, data_dict, is_valid=False, is_sdf=False):

        mesh = data_dict["mesh"]
        calib = data_dict["calib"]

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        vert_ids = np.arange(mesh.verts.shape[0])

        samples_surface_ids = np.random.choice(vert_ids,
                                               n_samples_surface,
                                               replace=True)

        samples_surface = mesh.verts[samples_surface_ids, :]

        # Sampling offsets are random noise with constant scale (15cm - 20cm)
        offset = np.random.normal(scale=self.opt.sigma_geo,
                                  size=(n_samples_surface, 1))
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
        labels = np.concatenate([
            np.ones(inside_samples.shape[0]),
            np.zeros(outside_samples.shape[0])
        ])

        samples = torch.from_numpy(samples).float()
        labels = torch.from_numpy(labels).float()

        return {"samples_geo": samples, "labels_geo": labels}

    def visualize_sampling3D(self, data_dict, mode="vis"):

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg="white")
        vis_list = []

        assert mode in ["vis", "sdf", "norm", "cmap", "occ", "kpt"]

        # sdf-1 cmap-3 norm-3 vis-1
        if mode == "vis":
            labels = data_dict[f"smpl_feat"][:, [-1]]  # visibility
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == "occ":
            labels = data_dict[f"labels_geo"][..., None]  # occupancy
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == "sdf":
            labels = data_dict[f"smpl_feat"][:, [0]]  # sdf
            labels -= labels.min()
            labels /= labels.max()
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == "norm":
            labels = data_dict[f"smpl_feat"][:, -4:-1]  # normal
            colors = (labels + 1.0) * 0.5
        elif mode == "cmap":
            labels = data_dict[f"smpl_feat"][:, -7:-4]  # colormap
            colors = np.array(labels)
        elif mode == "kpt":
            colors = np.ones_like(data_dict["smpl_joint"])

        points = projection(data_dict["samples_geo"], data_dict["calib"])
        verts = projection(data_dict["verts"], data_dict["calib"])
        points[:, 1] *= -1
        verts[:, 1] *= -1

        keypoints = data_dict["smpl_joint"]
        keypoints[:, 1] *= -1  # [-1,1] 24x3

        # create a mesh
        if mode == 'kpt':
            alpha = 0.2
        else:
            alpha = 1.0

        mesh = trimesh.Trimesh(verts, data_dict["faces"], process=True)
        mesh.visual.vertex_colors = [128.0, 128.0, 128.0, alpha * 255.0]
        vis_list.append(mesh)

        if "voxel_verts" in data_dict.keys():
            print(colored("voxel verts", "green"))
            voxel_verts = data_dict["voxel_verts"] * 2.0
            voxel_faces = data_dict["voxel_faces"]
            voxel_verts[:, 1] *= -1
            voxel = trimesh.Trimesh(
                voxel_verts,
                voxel_faces[:, [0, 2, 1]],
                process=False,
                maintain_order=True,
            )
            voxel.visual.vertex_colors = [0.0, 128.0, 0.0, alpha * 255.0]
            vis_list.append(voxel)

        if "smpl_verts" in data_dict.keys():
            print(colored("smpl verts", "green"))
            smplx_verts = data_dict["smpl_verts"]
            smplx_faces = data_dict["smpl_faces"]
            smplx_verts[:, 1] *= -1
            smplx = trimesh.Trimesh(
                smplx_verts,
                smplx_faces[:, [0, 2, 1]],
                process=False,
                maintain_order=True,
            )
            smplx.visual.vertex_colors = [128.0, 128.0, 0.0, alpha * 255.0]
            vis_list.append(smplx)

        if mode != 'kpt':
            # create a pointcloud
            pc = vedo.Points(points, r=15, c=np.float32(colors))
            vis_list.append(pc)

        # create a picure
        img_pos = [1.0, 0.0, -1.0]
        for img_id, img_key in enumerate(["normal_F", "image", "T_normal_B"]):
            image_arr = (
                (data_dict[img_key].detach().cpu().permute(1, 2, 0).numpy() +
                 1.0) * 0.5 * 255.0)
            image_dim = image_arr.shape[0]
            image = (vedo.Picture(image_arr).scale(2.0 / image_dim).pos(
                -1.0, -1.0, img_pos[img_id]))
            vis_list.append(image)

        # create skeleton
        kpt = vedo.Points(keypoints,
                          r=15,
                          c=np.float32(np.ones_like(keypoints)))
        vis_list.append(kpt)

        vp.show(*vis_list, bg="white", axes=1.0, interactive=True)
