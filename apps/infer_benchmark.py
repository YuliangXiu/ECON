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

import torch
import trimesh
import numpy as np
import argparse
import os

from termcolor import colored
from tqdm.auto import tqdm
from apps.Normal import Normal
from apps.IFGeo import IFGeo
from lib.common.config import cfg
from lib.common.render import Render
from lib.common.train_util import load_normal_networks, load_networks
from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.dataset.PIFuDataset import PIFuDataset
from lib.dataset.mesh_util import *
from lib.common.voxelize import VoxelGrid

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/bni.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    device = torch.device(f"cuda:{args.gpu_device}")

    cfg_test_list = [
        "test_mode",
        True,
        "dataset.types",
        ["renderpeople", "cape"],
        "dataset.scales",
        [1.0, 100.0],
        "dataset.rotation_num",
        3,
        "mcube_res",
        256,
        "clean_mesh",
        True,
        "batch_size",
        1,
    ]

    cfg.merge_from_list(cfg_test_list)
    cfg.freeze()

    # load model
    normal_model = Normal(cfg).to(device)
    load_normal_networks(normal_model, cfg.normal_path)

    # load IFGeo model
    ifnet_model = IFGeo(cfg).to(device)
    load_networks(ifnet_model, mlp_path=cfg.ifnet_path)

    normal_model.netG.eval()
    ifnet_model.netG.eval()

    # SMPLX object
    SMPLX_object = SMPLX()
    dataset = PIFuDataset(cfg=cfg, split="test")

    export_dir = osp.join(cfg.results_path, cfg.name, "-".join(cfg.dataset.types))

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:

        # dict_keys(['dataset', 'subject', 'rotation', 'scale',
        # 'smplx_param', 'smpl_param', 'calib',
        # 'image', 'T_normal_F', 'T_normal_B',
        # 'verts', 'faces',
        # 'smpl_vis', 'smpl_joint', 'smpl_verts', 'smpl_faces'])

        for key in data.keys():
            if torch.is_tensor(data[key]):
                data[key] = data[key].unsqueeze(0).to(device)

        current_name = data["dataset"] + "_" + data["subject"]
        pbar.set_description(current_name)

        current_dir = osp.join(export_dir, data['dataset'], data['subject'])
        os.makedirs(current_dir, exist_ok=True)

        in_tensor = data.copy()

        batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0]).to(device)
        batch_smpl_faces = in_tensor["smpl_faces"].detach()

        in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces)
        
        with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_model.netG(in_tensor)
                
        smpl_mesh = trimesh.Trimesh(batch_smpl_verts.cpu().numpy()[0],
                                    batch_smpl_faces.cpu().numpy()[0])

        side_mesh = smpl_mesh.copy()
        face_mesh = smpl_mesh.copy()
        hand_mesh = smpl_mesh.copy()

        # save normals, depths and masks
        BNI_dict = save_normal_tensor(
            in_tensor,
            0,
            osp.join(current_dir, "BNI/param_dict"),
        )

        # BNI process
        BNI_object = BNI(dir_path=osp.join(current_dir, "BNI"),
                         name=current_name,
                         BNI_dict=BNI_dict,
                         device=device,
                         mvc=False)

        BNI_object.extract_surface(0, False)

        if cfg.always_ifnet:

            side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

            # mesh completion via IF-net
            in_tensor.update(
                dataset.depth_to_voxel(
                    {
                        "depth_F": BNI_object.F_depth.unsqueeze(0).to(device),
                        "depth_B": BNI_object.B_depth.unsqueeze(0).to(device)
                    }, cfg.vol_res))

            occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                0,
            ] * 3, scale=2.0).data.transpose(2, 1, 0)
            occupancies = np.flip(occupancies, axis=1)

            in_tensor["body_voxels"] = torch.tensor(
                occupancies.copy()).float().unsqueeze(0).to(device)

            with torch.no_grad():
                sdf = ifnet_model.reconEngine(netG=ifnet_model.netG, batch=in_tensor)
                verts_IF, faces_IF = ifnet_model.reconEngine.export_mesh(sdf)

            if ifnet_model.clean_mesh_flag:
                verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)

            verts_IF -= (ifnet_model.resolutions[-1] - 1) / 2.0
            verts_IF /= (ifnet_model.resolutions[-1] - 1) / 2.0

            side_mesh = trimesh.Trimesh(verts_IF, faces_IF)
            side_mesh_path = osp.join(current_dir, f"{current_name}_IF.obj")
            side_mesh = remesh(side_mesh, side_mesh_path)

        else:
            print(colored("High overlap, use SMPL-X body\n", "green"))
            side_mesh = apply_vertex_mask(
                side_mesh,
                (SMPLX_object.front_flame_vertex_mask + SMPLX_object.mano_vertex_mask +
                 SMPLX_object.eyeball_vertex_mask).eq(0).float(),
            )

        side_verts = torch.tensor(side_mesh.vertices).float().to(device)
        side_faces = torch.tensor(side_mesh.faces).long().to(device)

        # Possion Fusion between SMPLX and BNI
        # 1. keep the faces invisible to front+back cameras
        # 2. keep the front-FLAME+MANO faces
        # 3. remove eyeball faces

        (xy, z) = side_verts.split([2, 1], dim=-1)
        F_vis = get_visibility(xy, z, side_faces[..., [0, 2, 1]], img_res=2**8)
        B_vis = get_visibility(xy, -z, side_faces, img_res=2**8)

        full_lst = []

        if "face" in cfg.use_smpl:
            # only face
            face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)
            face_mesh.vertices[:, 2] -= BNI_object.thickness.numpy() / 2.0
            # remove face neighbor triangles
            BNI_object.F_B_trimesh = part_removal(BNI_object.F_B_trimesh, None, face_mesh, 4e-2,
                                                  device)
            full_lst += [face_mesh]

        if "hand" in cfg.use_smpl:
            # only hands
            hand_mesh = apply_vertex_mask(hand_mesh, SMPLX_object.mano_vertex_mask)
            # remove face neighbor triangles
            BNI_object.F_B_trimesh = part_removal(BNI_object.F_B_trimesh, None, hand_mesh, 4e-2,
                                                  device)
            full_lst += [hand_mesh]

        full_lst += [BNI_object.F_B_trimesh]

        # initial side_mesh could be SMPLX or IF-net
        side_mesh = part_removal(side_mesh,
                                 torch.logical_or(F_vis, B_vis),
                                 sum(full_lst),
                                 1e-2,
                                 device,
                                 clean=False)

        full_lst += [side_mesh]

        # export intermediate meshes
        BNI_object.F_B_trimesh.export(osp.join(current_dir, f"{current_name}_F_B.obj"))

        side_mesh.export(osp.join(current_dir, f"{current_name}_side.obj"))

        final_mesh = poisson(
            sum(full_lst),
            osp.join(current_dir, f"{current_name}_final.obj"),
            np.log2(cfg.mcube_res),
        )