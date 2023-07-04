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

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os

import numpy as np
import torch
import torchvision
import trimesh
from pytorch3d.ops import SubdivideMeshes
from termcolor import colored
from tqdm.auto import tqdm

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm
from lib.common.local_affine import register
from lib.common.render import query_color
from lib.common.train_util import Format, init_loss
from lib.common.voxelize import VoxelGrid
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-novis", action="store_true")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymafx/configs/pymafx_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    # setting for testing on in-the-wild images
    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", True, "test_mode", True,
        "batch_size", 1
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    # load normal model
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg, checkpoint_path=cfg.normal_path, map_location=device, strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(
        colored(
            f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"
        )
    )

    # SMPLX object
    SMPLX_object = SMPLX()

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,    # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,    # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }

    if cfg.bni.use_ifnet:
        # load IFGeo model
        ifnet = IFGeo.load_from_checkpoint(
            cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False
        )
        ifnet = ifnet.to(device)
        ifnet.netG.eval()

        print(colored(f"Resume IF-Net+ from {Format.start} {cfg.ifnet_path} {Format.end}", "green"))
        print(colored(f"Complete with {Format.start} IF-Nets+ (Implicit) {Format.end}", "green"))
    else:
        print(colored(f"Complete with {Format.start} SMPL-X (Explicit) {Format.end}", "green"))

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:

        losses = init_loss()

        pbar.set_description(f"{data['name']}")

        # final results rendered as image (PNG)
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)
        # 4. Blend the cropped image with predicted cloth normal (xxx_crop.png)

        os.makedirs(osp.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes (OBJ)
        # 1. SMPL mesh (xxx_smpl_xx.obj)
        # 2. SMPL params (xxx_smpl.npy)
        # 3. d-BiNI surfaces (xxx_BNI.obj)
        # 4. seperate face/hand mesh (xxx_hand/face.obj)
        # 5. full shape impainted by IF-Nets+ after remeshing (xxx_IF.obj)
        # 6. sideded or occluded parts (xxx_side.obj)
        # 7. final reconstructed clothed human (xxx_full.obj)

        os.makedirs(osp.join(args.out_dir, cfg.name, "obj"), exist_ok=True)

        in_tensor = {
            "smpl_faces": data["smpl_faces"], "image": data["img_icon"].to(device), "mask":
            data["img_mask"].to(device)
        }

        # The optimizer and variables
        optimed_pose = data["body_pose"].requires_grad_(True)
        optimed_trans = data["trans"].requires_grad_(True)
        optimed_betas = data["betas"].requires_grad_(True)
        optimed_orient = data["global_orient"].requires_grad_(True)

        optimizer_smpl = torch.optim.Adam([
            optimed_pose, optimed_trans, optimed_betas, optimed_orient
        ],
                                          lr=1e-2,
                                          amsgrad=True)
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        N_body, N_pose = optimed_pose.shape[:2]

        smpl_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_00.obj"

        # remove this line if you change the loop_smpl and obtain different SMPL-X fits
        if osp.exists(smpl_path):

            smpl_verts_lst = []
            smpl_faces_lst = []

            for idx in range(N_body):

                smpl_obj = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"
                smpl_mesh = trimesh.load(smpl_obj)
                smpl_verts = torch.tensor(smpl_mesh.vertices).to(device).float()
                smpl_faces = torch.tensor(smpl_mesh.faces).to(device).long()
                smpl_verts_lst.append(smpl_verts)
                smpl_faces_lst.append(smpl_faces)

            batch_smpl_verts = torch.stack(smpl_verts_lst)
            batch_smpl_faces = torch.stack(smpl_faces_lst)

            # render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                batch_smpl_verts, batch_smpl_faces
            )

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

            in_tensor["smpl_verts"] = batch_smpl_verts * torch.tensor([1., -1., 1.]).to(device)
            in_tensor["smpl_faces"] = batch_smpl_faces[:, :, [0, 2, 1]]

        else:
            # smpl optimization
            loop_smpl = tqdm(range(args.loop_smpl))

            for i in loop_smpl:

                per_loop_lst = []

                optimizer_smpl.zero_grad()

                N_body, N_pose = optimed_pose.shape[:2]

                # 6d_rot to rot_mat
                optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1,
                                                                         6)).view(N_body, 1, 3, 3)
                optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1,
                                                                     6)).view(N_body, N_pose, 3, 3)

                smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=optimed_pose_mat,
                    global_pose=optimed_orient_mat,
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data["right_hand_pose"], device),
                )

                smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor([
                    1.0, 1.0, -1.0
                ]).to(device)

                # landmark errors
                smpl_joints_3d = (
                    smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
                ) * 0.5
                in_tensor["smpl_joint"] = smpl_joints[:,
                                                      dataset.smpl_data.smpl_joint_ids_24_pixie, :]

                ghum_lmks = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                ghum_conf = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

                # render optimized mesh as normal [-1,1]
                in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                    smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    in_tensor["smpl_faces"],
                )

                T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

                with torch.no_grad():
                    in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

                diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
                diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

                # silhouette loss
                smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
                gt_arr = in_tensor["mask"].repeat(1, 1, 2)
                diff_S = torch.abs(smpl_arr - gt_arr)
                losses["silhouette"]["value"] = diff_S.mean()

                # large cloth_overlap --> big difference between body and cloth mask
                # for loose clothing, reply more on landmarks instead of silhouette+normal loss
                cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
                cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
                losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]

                # small body_overlap --> large occlusion or out-of-frame
                # for highly occluded body, reply only on high-confidence landmarks, no silhouette+normal loss

                # BUG: PyTorch3D silhouette renderer generates dilated mask
                bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
                smpl_arr_fake = torch.cat([
                    in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                    in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
                ],
                                          dim=-1)

                body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                               ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
                body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
                body_overlap_flag = body_overlap < cfg.body_overlap_thres

                losses["normal"]["value"] = (
                    diff_F_smpl * body_overlap_mask[..., :512] +
                    diff_B_smpl * body_overlap_mask[..., 512:]
                ).mean() / 2.0

                losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]
                occluded_idx = torch.where(body_overlap_flag)[0]
                ghum_conf[occluded_idx] *= ghum_conf[occluded_idx] > 0.95
                losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) *
                                            ghum_conf).mean(dim=1)

                # Weighted sum of the losses
                smpl_loss = 0.0
                pbar_desc = "Body Fitting -- "
                for k in ["normal", "silhouette", "joint"]:
                    per_loop_loss = (
                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                    ).mean()
                    pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                    smpl_loss += per_loop_loss
                pbar_desc += f"Total: {smpl_loss:.3f}"
                loose_str = ''.join([str(j) for j in cloth_overlap_flag.int().tolist()])
                occlude_str = ''.join([str(j) for j in body_overlap_flag.int().tolist()])
                pbar_desc += colored(f"| loose:{loose_str}, occluded:{occlude_str}", "yellow")
                loop_smpl.set_description(pbar_desc)

                # save intermediate results
                if (i == args.loop_smpl - 1) and (not args.novis):

                    per_loop_lst.extend([
                        in_tensor["image"],
                        in_tensor["T_normal_F"],
                        in_tensor["normal_F"],
                        diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_loop_lst.extend([
                        in_tensor["image"],
                        in_tensor["T_normal_B"],
                        in_tensor["normal_B"],
                        diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
                    ])
                    per_data_lst.append(
                        get_optim_grid_image(per_loop_lst, None, nrow=N_body * 2, type="smpl")
                    )

                smpl_loss.backward()
                optimizer_smpl.step()
                scheduler_smpl.step(smpl_loss)

            in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)
            in_tensor["smpl_faces"] = in_tensor["smpl_faces"][:, :, [0, 2, 1]]

            if not args.novis:
                per_data_lst[-1].save(
                    osp.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png")
                )

        if not args.novis:
            img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_crop.png")
            torchvision.utils.save_image(
                torch.cat([
                    data["img_crop"][:, :3], (in_tensor['normal_F'].detach().cpu() + 1.0) * 0.5,
                    (in_tensor['normal_B'].detach().cpu() + 1.0) * 0.5
                ],
                          dim=3), img_crop_path
            )

            rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
            rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)

            img_overlap_path = osp.join(args.out_dir, cfg.name, f"png/{data['name']}_overlap.png")
            torchvision.utils.save_image(
                torch.cat([data["img_raw"], rgb_norm_F, rgb_norm_B], dim=-1) / 255.,
                img_overlap_path
            )

        smpl_obj_lst = []

        for idx in range(N_body):

            smpl_obj = trimesh.Trimesh(
                in_tensor["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                process=False,
                maintains_order=True,
            )

            smpl_obj_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"

            if not osp.exists(smpl_obj_path):
                smpl_obj.export(smpl_obj_path)
                smpl_info = {
                    "betas":
                    optimed_betas[idx].detach().cpu().unsqueeze(0),
                    "body_pose":
                    rotation_matrix_to_angle_axis(optimed_pose_mat[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "global_orient":
                    rotation_matrix_to_angle_axis(optimed_orient_mat[idx].detach()
                                                 ).cpu().unsqueeze(0),
                    "transl":
                    optimed_trans[idx].detach().cpu(),
                    "expression":
                    data["exp"][idx].cpu().unsqueeze(0),
                    "jaw_pose":
                    rotation_matrix_to_angle_axis(data["jaw_pose"][idx]).cpu().unsqueeze(0),
                    "left_hand_pose":
                    rotation_matrix_to_angle_axis(data["left_hand_pose"][idx]).cpu().unsqueeze(0),
                    "right_hand_pose":
                    rotation_matrix_to_angle_axis(data["right_hand_pose"][idx]).cpu().unsqueeze(0),
                    "scale":
                    data["scale"][idx].cpu(),
                }
                np.save(
                    smpl_obj_path.replace(".obj", ".npy"),
                    smpl_info,
                    allow_pickle=True,
                )
            smpl_obj_lst.append(smpl_obj)

        del optimizer_smpl
        del optimed_betas
        del optimed_orient
        del optimed_pose
        del optimed_trans

        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement

        per_data_lst = []

        batch_smpl_verts = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0],
                                                                           device=device)
        batch_smpl_faces = in_tensor["smpl_faces"].detach()[:, :, [0, 2, 1]]

        in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces
        )

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []

        for idx in range(N_body):

            final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()
            smplx_mesh = smpl_obj_lst[idx].copy()

            # save normals, depths and masks
            BNI_dict = save_normal_tensor(
                in_tensor,
                idx,
                osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
                cfg.bni.thickness,
            )

            # BNI process
            BNI_object = BNI(
                dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                name=data["name"],
                BNI_dict=BNI_dict,
                cfg=cfg.bni,
                device=device
            )

            BNI_object.extract_surface(False)

            in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
            in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh

            if cfg.bni.use_ifnet:

                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"

                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

                # mesh completion via IF-net
                in_tensor.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0), "depth_B":
                        BNI_object.B_depth.unsqueeze(0)
                    })
                )

                occupancies = VoxelGrid.from_mesh(side_mesh, cfg.vol_res, loc=[
                    0,
                ] * 3, scale=2.0).data.transpose(2, 1, 0)
                occupancies = np.flip(occupancies, axis=1)

                in_tensor["body_voxels"] = torch.tensor(occupancies.copy()
                                                       ).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    sdf = ifnet.reconEngine(netG=ifnet.netG, batch=in_tensor)
                    verts_IF, faces_IF = ifnet.reconEngine.export_mesh(sdf)

                if ifnet.clean_mesh_flag:
                    verts_IF, faces_IF = clean_mesh(verts_IF, faces_IF)

                side_mesh = trimesh.Trimesh(verts_IF, faces_IF)
                side_mesh = remesh_laplacian(side_mesh, side_mesh_path)

            else:
                side_mesh = apply_vertex_mask(
                    side_mesh,
                    (
                        SMPLX_object.front_flame_vertex_mask + SMPLX_object.smplx_mano_vertex_mask +
                        SMPLX_object.eyeball_vertex_mask
                    ).eq(0).float(),
                )

                #register side_mesh to BNI surfaces
                side_mesh = Meshes(
                    verts=[torch.tensor(side_mesh.vertices).float()],
                    faces=[torch.tensor(side_mesh.faces).long()],
                ).to(device)
                sm = SubdivideMeshes(side_mesh)
                side_mesh = register(BNI_object.F_B_trimesh, sm(side_mesh), device)

            side_verts = torch.tensor(side_mesh.vertices).float().to(device)
            side_faces = torch.tensor(side_mesh.faces).long().to(device)

            # Possion Fusion between SMPLX and BNI
            # 1. keep the faces invisible to front+back cameras
            # 2. keep the front-FLAME+MANO faces
            # 3. remove eyeball faces

            # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
            full_lst = []

            if "face" in cfg.bni.use_smpl:

                # only face
                face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)

                if not face_mesh.is_empty:
                    face_mesh.vertices = face_mesh.vertices - np.array([0, 0, cfg.bni.thickness])

                    # remove face neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(
                        BNI_object.F_B_trimesh,
                        face_mesh,
                        cfg.bni.face_thres,
                        device,
                        smplx_mesh,
                        region="face"
                    )
                    side_mesh = part_removal(
                        side_mesh, face_mesh, cfg.bni.face_thres, device, smplx_mesh, region="face"
                    )
                    face_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_face.obj")
                    full_lst += [face_mesh]

            if "hand" in cfg.bni.use_smpl:
                hand_mask = torch.zeros(SMPLX_object.smplx_verts.shape[0], )

                if data['hands_visibility'][idx][0]:

                    mano_left_vid = np.unique(
                        np.concatenate([
                            SMPLX_object.smplx_vert_seg["leftHand"],
                            SMPLX_object.smplx_vert_seg["leftHandIndex1"],
                        ])
                    )

                    hand_mask.index_fill_(0, torch.tensor(mano_left_vid), 1.0)

                if data['hands_visibility'][idx][1]:

                    mano_right_vid = np.unique(
                        np.concatenate([
                            SMPLX_object.smplx_vert_seg["rightHand"],
                            SMPLX_object.smplx_vert_seg["rightHandIndex1"],
                        ])
                    )

                    hand_mask.index_fill_(0, torch.tensor(mano_right_vid), 1.0)

                # only hands
                hand_mesh = apply_vertex_mask(hand_mesh, hand_mask)

                if not hand_mesh.is_empty:
                    # remove hand neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(
                        BNI_object.F_B_trimesh,
                        hand_mesh,
                        cfg.bni.hand_thres,
                        device,
                        smplx_mesh,
                        region="hand"
                    )
                    side_mesh = part_removal(
                        side_mesh, hand_mesh, cfg.bni.hand_thres, device, smplx_mesh, region="hand"
                    )
                    hand_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_hand.obj")
                    full_lst += [hand_mesh]

            full_lst += [BNI_object.F_B_trimesh]

            # initial side_mesh could be SMPLX or IF-net
            side_mesh = part_removal(
                side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
            )

            full_lst += [side_mesh]

            # # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj"
            )
            side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

            if cfg.bni.use_poisson:
                final_mesh = poisson(
                    sum(full_lst),
                    final_path,
                    cfg.bni.poisson_depth,
                )
                print(
                    colored(
                        f"\n Poisson completion to {Format.start} {final_path} {Format.end}",
                        "yellow"
                    )
                )
            else:
                final_mesh = sum(full_lst)
                final_mesh.export(final_path)

            if not args.novis:
                dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
                rotate_recon_lst = dataset.render.get_image(cam_type="four")
                per_loop_lst.extend([in_tensor['image'][idx:idx + 1]] + rotate_recon_lst)

            if cfg.bni.texture_src == 'image':

                # coloring the final mesh (front: RGB pixels, back: normal colors)
                final_colors = query_color(
                    torch.tensor(final_mesh.vertices).float(),
                    torch.tensor(final_mesh.faces).long(),
                    in_tensor["image"][idx:idx + 1],
                    device=device,
                )
                final_mesh.visual.vertex_colors = final_colors
                final_mesh.export(final_path)

            elif cfg.bni.texture_src == 'SD':

                # !TODO: add texture from Stable Diffusion
                pass

        if len(per_loop_lst) > 0 and (not args.novis):

            per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))
            per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))

            # for video rendering
            in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

            os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
            in_tensor["uncrop_param"] = data["uncrop_param"]
            in_tensor["img_raw"] = data["img_raw"]
            torch.save(
                in_tensor, osp.join(args.out_dir, cfg.name, f"vid/{data['name']}_in_tensor.pt")
            )
