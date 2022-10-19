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


class IFDataset:

    def __init__(self, cfg, split="train", vis=False, cached=True):

        self.split = split
        self.root = cfg.root
        self.bsize = cfg.batch_size
        self.overfit = cfg.overfit

        # for debug, only used in visualize_sampling3D
        self.vis = vis
        if self.vis:
            self.current_epoch = 0
        else:
            # self.current_epoch = self.trainer.current_epoch
            self.current_epoch = 0

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.img_res = self.opt.input_size
        self.vol_res = self.opt.voxel_res
        self.scales = self.opt.scales
        self.prior_type = self.opt.prior_type
        self.noise_type = self.opt.noise_type
        self.noise_scale = self.opt.noise_scale
        self.sigma_geo = self.opt.sigma_geo / 5.0 if self.current_epoch >= cfg.schedule[
            0] else self.opt.sigma_geo

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

        if self.split == "train":
            self.rotations = np.arange(0, 360, 360 / self.opt.rotation_num).astype(np.int32)
        elif self.split == "test":
            self.rotations = range(0, 360, 120)
        else:
            self.rotations = [0]

        self.datasets_dict = {}
        self.mesh_cached = {}

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

            self.datasets_dict[dataset].update(
                {"subjects": np.loadtxt(osp.join(dataset_dir, f"{split}.txt"), dtype=str)})

            # pre-cached meshes
            if cached:
                self.mesh_cached[dataset] = {}
                pbar = tqdm(self.datasets_dict[dataset]["subjects"])
                for subject in pbar:
                    subject = subject.split("/")[-1]
                    pbar.set_description(f"Loading {dataset}-{split}-{subject}")
                    if dataset == "thuman2":
                        mesh_path = osp.join(self.datasets_dict[dataset]["mesh_dir"],
                                            f"{subject}/{subject}.obj")
                    else:
                        mesh_path = osp.join(self.datasets_dict[dataset]["mesh_dir"], f"{subject}.obj")

                    if subject not in self.mesh_cached[dataset].keys():
                        self.mesh_cached[dataset][subject] = self.load_mesh(
                            mesh_path, self.scales[dataset_id])

        self.subject_list = self.get_subject_list(split)
        self.smplx = SMPLX()

    def transform_to_tensor(self, res, mean, std):
        return transforms.Compose([
            transforms.Resize(res),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            split_txt = osp.join(self.root, dataset, f"{split}.txt")

            if osp.exists(split_txt) and osp.getsize(split_txt) > 0:
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        if self.split == "train":
            subject_list += subject_list[:self.bsize - len(subject_list) % self.bsize]
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
        if self.vis:
            data_dict.update(self.load_smpl(data_dict, self.vis))
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

    def load_calib(self, data_dict):
        calib_data = np.loadtxt(data_dict["calib_path"], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {"calib": calib_mat}

    def depth_to_voxel(self, data_dict):

        depth_mask = (data_dict['depth_F'] != 0.).float()
        depth_mask = kornia.augmentation.RandomErasing(p=1.0,
                                                       scale=(0.01, 0.2),
                                                       ratio=(0.3, 3.3),
                                                       keepdim=True)(depth_mask)

        depth_FB = torch.cat([data_dict['depth_F'], -data_dict['depth_B']], dim=0) * depth_mask
        index_z = ((depth_FB + 1.) * 0.5 * self.vol_res).round().clip(0, self.vol_res -
                                                                      1).long().permute(1, 2, 0)
        index_mask = index_z[..., 0] == torch.tensor(self.vol_res * 0.5).long()
        if index_z.max() >= self.vol_res - 1:
            print(data_dict["subject"])
        voxels = F.one_hot(index_z[..., 0], self.vol_res) + F.one_hot(index_z[..., 1], self.vol_res)
        voxels[index_mask] *= 0
        voxels = torch.flip(voxels, [2]).permute(2, 0, 1).float()  #[x-2, y-0, z-1]
        return {"depth_voxels": voxels, "depth_mask": depth_mask}

    @staticmethod
    def load_mesh(mesh_path, scale):

        verts, faces, _, _ = obj_loader(mesh_path)

        mesh = HoppeMesh(verts * scale, faces)

        return mesh

    def add_noise(self, beta_num, smpl_pose, smpl_betas, noise_type, noise_scale, type, hashcode):

        # np.random.seed(hashcode)

        if type == "smplx":
            noise_idx = self.noise_smplx_idx
        else:
            noise_idx = self.noise_smpl_idx

        if "beta" in noise_type and noise_scale[noise_type.index("beta")] > 0.0:
            smpl_betas += ((np.random.rand(beta_num) - 0.5) * 2.0 *
                           noise_scale[noise_type.index("beta")])
            smpl_betas = smpl_betas.astype(np.float32)

        if "pose" in noise_type and noise_scale[noise_type.index("pose")] > 0.0:
            smpl_pose[noise_idx] += ((np.random.rand(len(noise_idx)) - 0.5) * 2.0 * np.pi *
                                     noise_scale[noise_type.index("pose")])
            smpl_pose = smpl_pose.astype(np.float32)
        if type == "smplx":
            return torch.as_tensor(smpl_pose[None, ...]), torch.as_tensor(smpl_betas[None, ...])
        else:
            return smpl_pose, smpl_betas

    def compute_smpl_verts(self, data_dict, noise_type=None, noise_scale=None, smpl_type='smplx'):

        dataset = data_dict["dataset"]
        smplx_dict = {}

        smplx_param = np.load(data_dict[f"{smpl_type}_param"], allow_pickle=True)
        smplx_pose = smplx_param["body_pose"]  # [1,63]
        smplx_betas = smplx_param["betas"]  # [1,10]
        smplx_pose, smplx_betas = self.add_noise(
            smplx_betas.shape[1],
            smplx_pose[0],
            smplx_betas[0],
            noise_type,
            noise_scale,
            type=smpl_type,
            hashcode=(hash(f"{data_dict['subject']}_{data_dict['rotation']}")) % (10**8),
        )

        smplx_out, smplx_joints = load_fit_body(
            fitted_path=data_dict[f"{smpl_type}_param"],
            scale=self.datasets_dict[dataset]["scale"],
            smpl_type=smpl_type,
            smpl_gender="male",
            noise_dict=dict(betas=smplx_betas, body_pose=smplx_pose),
        )

        smplx_dict.update({
            "smpl_joint": torch.as_tensor(smplx_joints).float(),
            "smpl_verts": torch.as_tensor(smplx_out.vertices).float(),
        })

        return smplx_dict

    def compute_voxel_verts(self, data_dict, noise_type=None, noise_scale=None):

        from lib.pymaf.utils.geometry import rotation_matrix_to_angle_axis
        from lib.dataset.body_model import TetraSMPLModel

        smpl_param = np.load(data_dict["smpl_param"], allow_pickle=True)

        if data_dict['dataset'] == "thuman2":
            gender = 'male'
            smpl_trans = smpl_param["transl"]
        else:
            gender = np.load(data_dict['smplx_param'], allow_pickle=True)['gender']
            smpl_trans = smpl_param["translation"][0].numpy()

        smpl_pose = rotation_matrix_to_angle_axis(torch.as_tensor(
            smpl_param["full_pose"][0])).numpy()
        smpl_betas = smpl_param["betas"].flatten()

        if smpl_betas.shape[0] == 11:
            age = 'kid'
        else:
            age = 'adult'

        smpl_path = osp.join(self.smplx.model_dir, f"smpl/SMPL_{gender.upper()}.pkl")
        tetra_path = osp.join(self.smplx.tedra_dir, f"tetra_{gender}_{age}_smpl.npz")

        smpl_model = TetraSMPLModel(smpl_path, tetra_path, age)

        if sum(self.noise_scale) > 0.0:
            smpl_pose, smpl_betas = self.add_noise(
                smpl_model.beta_shape[0],
                smpl_pose.flatten(),
                smpl_betas[0],
                noise_type,
                noise_scale,
                type="smpl",
                hashcode=(hash(f"{data_dict['subject']}_{data_dict['rotation']}")) % (10**8),
            )

        smpl_model.set_params(pose=smpl_pose.reshape(-1, 3), beta=smpl_betas, trans=smpl_trans)

        if data_dict['dataset'] == "thuman2":
            verts = (np.concatenate([smpl_model.verts, smpl_model.verts_added], axis=0) *
                     smpl_param["scale"] +
                     smpl_param["translation"]) * self.datasets_dict[data_dict["dataset"]]["scale"]
        else:
            verts = np.concatenate([smpl_model.verts, smpl_model.verts_added], axis=0) * 100.0

        faces = np.loadtxt(
            osp.join(self.smplx.tedra_dir, f"tetrahedrons_{gender}_{age}.txt"),
            dtype=np.int32,
        ) - 1

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = np.pad(verts, ((0, pad_v_num), (0, 0)), mode="constant",
                       constant_values=0.0).astype(np.float32)
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)), mode="constant",
                       constant_values=0.0).astype(np.int32)

        return_dict = {
            "voxel_verts": projection(verts, data_dict["calib"]) * 0.5,
            "voxel_faces": torch.tensor(faces),
            "voxel_pad_v_num": pad_v_num,
            "voxel_pad_f_num": pad_f_num,
        }

        if self.vis:
            return_dict.update({"voxelize_params": read_smpl_constants(self.smplx.tedra_dir)})

        return return_dict

    def load_smpl(self, data_dict, vis=False):

        smpl_type = ("smplx" if ("smplx_path" in data_dict.keys() and
                                 os.path.exists(data_dict["smplx_path"])) else "smpl")

        return_dict = {}

        # add random noise to SMPL-(X) params
        if sum(self.noise_scale) > 0.0:
            return_dict.update(
                self.compute_smpl_verts(data_dict, self.noise_type, self.noise_scale, smpl_type))

        # instead, directly load SMPL-(X) objs
        else:
            smplx_joints = (torch.as_tensor(np.load(data_dict["joint_path"])).float() * 100.0)
            smplx_verts = rescale_smpl(data_dict[f"{smpl_type}_path"], scale=100.0)
            return_dict.update({"smpl_joint": smplx_joints, "smpl_verts": smplx_verts})

        # skeleton loading
        smplx_joints = projection(return_dict["smpl_joint"], data_dict["calib"]).float()

        if smpl_type == "smplx":
            return_dict.update({"smpl_joint": smplx_joints[self.smpl_joint_ids, :]})
        else:
            return_dict.update({"smpl_joint": smplx_joints[:24, :]})

        return_dict["smpl_verts"] = projection(return_dict["smpl_verts"],
                                               data_dict["calib"]).float()
        return_dict["smpl_faces"] = torch.as_tensor(getattr(self.smplx,
                                                            f"{smpl_type}_faces")).long()

        return return_dict

    def get_sampling_geo(self, data_dict):

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
            labels = data_dict["labels_geo"][..., None]  # occupancy
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == "kpt":
            colors = np.ones_like(data_dict["smpl_joint"])

        points = projection(data_dict["samples_geo"], data_dict["calib"])
        verts = projection(self.mesh_cached[data_dict['dataset']][data_dict['subject']].verts,
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

        mesh = trimesh.Trimesh(verts,
                               self.mesh_cached[data_dict['dataset']][data_dict['subject']].faces,
                               process=True)
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

            # voxelization
            from lib.net.voxelize import Voxelization
            voxel_param = read_smpl_constants(self.smplx.tedra_dir)

            self.voxelization = Voxelization(
                voxel_param["smpl_vertex_code"],
                voxel_param["smpl_face_code"],
                voxel_param["smpl_faces"],
                voxel_param["smpl_tetras"],
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=1,
                device=torch.device("cuda:0"),
            )

            self.voxelization.update_param(voxel_faces)
            voxel_verts[:, 1] *= -1
            vol = self.voxelization(voxel_verts.unsqueeze(0).to(torch.device("cuda:0")) / 2.0)
            vis_list.append(vedo.Volume(vol[0, 0].detach().cpu().numpy()))

        if "depth_voxels" in data_dict.keys():
            depth_vol = vedo.Volume(data_dict["depth_voxels"].numpy())
            vis_list.append(depth_vol)

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
    dataset = IFDataset(cfg, "one", True)
    for data_dict in tqdm(dataset):
        # pass
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
        dataset.visualize_sampling3D(data_dict, mode="occ")
        # break
