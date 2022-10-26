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

import torch, torchvision
import trimesh
import numpy as np
import argparse
import pickle
import os

from termcolor import colored
from tqdm.auto import tqdm
from apps.Normal import Normal
from apps.IFGeo import IFGeo
from lib.common.cloth_extraction import extract_cloth
from lib.common.config import cfg
from lib.common.train_util import init_loss, load_normal_networks, load_networks
from lib.common.BNI import BNI
from lib.dataset.TestDataset import TestDataset
from lib.dataset.mesh_util import *
from lib.common.voxelize import VoxelGrid

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pixie")
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-BNI", action="store_false")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/bni.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymaf/configs/pymaf_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    # setting for testing on in-the-wild images
    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", False, "test_mode", True,
        "batch_size", 1
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    # load model
    normal_model = Normal(cfg).to(device)
    load_normal_networks(normal_model, cfg.normal_path)

    # load IFGeo model
    ifnet_model = IFGeo(cfg, device).to(device)
    load_networks(ifnet_model, mlp_path=cfg.ifnet_path)

    normal_model.netG.eval()
    ifnet_model.netG.eval()

    # SMPLX object
    SMPLX_object = SMPLX()

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,  # w/ or w/o segmentation
        "hps_type": args.hps_type,  # pymaf/pare/pixie
        "vol_res": cfg.vol_res,
    }

    if args.hps_type == "pixie" and "pamir" in args.config:
        print(colored("PIXIE isn't compatible with PaMIR, thus switch to PyMAF", "red"))
        dataset_param["hps_type"] = "pymaf"

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:
        losses = init_loss()

        pbar.set_description(f"{data['name']}")

        # final results rendered as image
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)

        os.makedirs(os.path.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes
        # 1. SMPL mesh (xxx_smpl.obj)
        # 2. SMPL params (xxx_smpl.npy)
        # 3. clohted mesh (xxx_recon.obj)
        # 4. remeshed clothed mesh (xxx_remesh.obj)
        # 5. refined clothed mesh (xxx_refine.obj)
        # 6. BNI clothed mesh (xxx_combine.obj)

        os.makedirs(os.path.join(args.out_dir, cfg.name, "obj"), exist_ok=True)

        img_crop_path = os.path.join(args.out_dir, cfg.name, "png", f"{data['name']}_crop.png")
        torchvision.utils.save_image(data["img_crop"], img_crop_path)

        in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}

        # The optimizer and variables
        optimed_pose = data["body_pose"].requires_grad_(True)
        optimed_trans = data["trans"].requires_grad_(True)
        optimed_betas = data["betas"].requires_grad_(True)
        optimed_orient = data["global_orient"].requires_grad_(True)

        optimizer_smpl = torch.optim.Adam(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient], lr=1e-2, amsgrad=True)
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        # smpl optimization
        loop_smpl = tqdm(range(args.loop_smpl if cfg.net.prior_type != "pifu" else 1))

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        for i in loop_smpl:

            per_loop_lst = []

            optimizer_smpl.zero_grad()

            N_body, N_pose = optimed_pose.shape[:2]

            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(N_body, N_pose, 3, 3)

            if dataset_param["hps_type"] != "pixie":
                smpl_out = dataset.smpl_model(
                    betas=optimed_betas,
                    body_pose=optimed_pose_mat,
                    global_orient=optimed_orient_mat,
                    transl=optimed_trans,
                    pose2rot=False,
                )

                smpl_verts = smpl_out.vertices * data["scale"]
                smpl_joints = smpl_out.joints * data["scale"]

            else:

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
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"]

            smpl_joints *= torch.tensor([1.0, 1.0, -1.0]).to(device)

            # landmark errors
            if data["type"] == "smpl":
                smpl_joints_3d = (smpl_joints[0, :, :] + 1.0) * 0.5
                in_tensor["smpl_joint"] = smpl_joints[:, :24, :]
            elif data["type"] == "smplx" and dataset_param["hps_type"] != "pixie":
                smpl_joints_3d = (smpl_joints[0, dataset.smpl_data.smpl_joint_ids_45, :] +
                                  1.0) * 0.5
                in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_data.smpl_joint_ids_24, :]
            else:
                smpl_joints_3d = (smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] +
                                  1.0) * 0.5
                in_tensor["smpl_joint"] = smpl_joints[:,
                                                      dataset.smpl_data.smpl_joint_ids_24_pixie, :]

            ghum_lmks = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]
            losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) *
                                        ghum_conf).mean(dim=1)

            # render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                in_tensor["smpl_faces"],
            )

            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_model.netG(in_tensor)

            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

            losses["normal"]["value"] = (diff_F_smpl + diff_B_smpl).mean() / 2.0

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
            gt_arr = torch.cat([in_tensor["normal_F"], in_tensor["normal_B"]],
                               dim=-1).permute(0, 2, 3, 1)
            gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
            bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
            gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()

            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()

            # for loose clothing, reply more on landmarks
            cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
            body_overlap = (gt_arr *
                            (smpl_arr > 0)).sum(dim=[1, 2]) / (smpl_arr > 0.).sum(dim=[1, 2])

            cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
            losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]

            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = "Body Fitting --- "
            for k in ["normal", "silhouette", "joint"]:
                per_loop_loss = (losses[k]["value"] *
                                 torch.tensor(losses[k]["weight"]).to(device)).mean()
                pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                smpl_loss += per_loop_loss
            pbar_desc += f"Total: {smpl_loss:.3f}"
            loop_smpl.set_description(pbar_desc)

            # save intermediate results / vis_freq and final_step

            if (i % args.vis_freq == 0) or (i == args.loop_smpl - 1):

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
                    get_optim_grid_image(per_loop_lst, None, nrow=N_body, type="smpl"))

            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)
            in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)

        # visualize the optimization process
        # 1. SMPL Fitting

        per_data_lst[-1].save(os.path.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png"))

        rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
        rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)

        img_overlap_path = os.path.join(args.out_dir, cfg.name, f"png/{data['name']}_overlap.png")
        torchvision.utils.save_image(
            torch.Tensor([data["img_raw"], rgb_norm_F, rgb_norm_B]).permute(0, 3, 1, 2) / 255.,
            img_overlap_path)

        smpl_obj_lst = []
        for idx in range(N_body):

            smpl_obj = trimesh.Trimesh(
                in_tensor["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0]),
                in_tensor["smpl_faces"].detach().cpu()[0],
                process=False,
                maintains_order=True,
            )
            smpl_obj.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj")
            smpl_obj_lst.append(smpl_obj)

        smpl_info = {
            "betas": optimed_betas,
            "pose": optimed_pose,
            "orient": optimed_orient,
            "trans": optimed_trans,
        }

        np.save(
            f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl.npy",
            smpl_info,
            allow_pickle=True,
        )

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
        batch_smpl_faces = in_tensor["smpl_faces"].detach()

        in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
            batch_smpl_verts, batch_smpl_faces)

        per_loop_lst = []

        in_tensor["BNI_verts"] = []
        in_tensor["BNI_faces"] = []
        in_tensor["body_verts"] = []
        in_tensor["body_faces"] = []

        pbar_body = tqdm(range(N_body))

        for idx in pbar_body:

            pbar_body.set_description(f"Body {idx:02d}")

            side_mesh = smpl_obj_lst[idx].copy()
            face_mesh = smpl_obj_lst[idx].copy()
            hand_mesh = smpl_obj_lst[idx].copy()

            # save normals, depths and masks
            BNI_dict = save_normal_tensor(
                in_tensor,
                idx,
                os.path.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
            )

            # BNI process
            BNI_object = BNI(dir_path=os.path.join(args.out_dir, cfg.name, "BNI"),
                             name=data["name"],
                             BNI_dict=BNI_dict,
                             device=device,
                             mvc=False)

            BNI_object.extract_surface(idx)

            side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)
            side_verts = torch.tensor(side_mesh.vertices).float()
            side_faces = torch.tensor(side_mesh.faces).long()

            in_tensor["body_verts"].append(side_verts)
            in_tensor["body_faces"].append(side_faces)

            # requires shape completion when low overlap
            # replace SMPL by completed mesh as side_mesh

            if body_overlap[idx] < cfg.body_overlap_thres:

                print(colored(f"Low overlap: {body_overlap[idx]:.2f}, shape completion\n", "green"))

                # mesh completion via IF-net
                in_tensor.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0),
                        "depth_B": BNI_object.B_depth.unsqueeze(0)
                    }))

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
                side_mesh_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_IF.obj"
                side_mesh = remesh(side_mesh, side_mesh_path)

                side_verts = torch.tensor(side_mesh.vertices).float()
                side_faces = torch.tensor(side_mesh.faces).long()
            else:
                print(colored("High overlap, use SMPL-X body\n", "green"))

            # Possion Fusion between SMPLX and BNI
            # 1. keep the faces invisible to front+back cameras
            # 2. keep the front-FLAME+MANO faces
            # 3. remove eyeball faces

            side_verts = side_verts.to(device)
            side_faces = side_faces.to(device)

            (xy, z) = side_verts.split([2, 1], dim=-1)
            F_vis = get_visibility(xy, z, side_faces[..., [0, 2, 1]], img_res=2**8)
            B_vis = get_visibility(xy, -z, side_faces, img_res=2**8)

            side_mesh = overlap_removal(side_mesh, torch.logical_or(F_vis, B_vis),
                                        BNI_object.F_B_trimesh, device)

            # only face
            front_flame_vertex_mask = torch.zeros(face_mesh.vertices.shape[0],)
            front_flame_vertex_mask[SMPLX_object.smplx_front_flame_vid] = 1.0
            face_mesh = apply_vertex_mask(face_mesh, front_flame_vertex_mask)

            # only hands
            mano_vertex_mask = torch.zeros(hand_mesh.vertices.shape[0],)
            mano_vertex_mask[SMPLX_object.smplx_mano_vid] = 1.0
            hand_mesh = apply_vertex_mask(hand_mesh, mano_vertex_mask)

            # remove hand/face neighbor triangles
            BNI_object.F_B_trimesh = face_hand_removal(BNI_object.F_B_trimesh, hand_mesh, face_mesh,
                                                       device)

            full_lst = [BNI_object.F_B_trimesh]

            if body_overlap[idx] < cfg.body_overlap_thres:
                side_mesh = face_hand_removal(side_mesh, hand_mesh, face_mesh, device)

            full_lst += [side_mesh, hand_mesh, face_mesh]

            # export intermediate meshes
            BNI_object.F_B_trimesh.export(
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj")

            side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

            sum([hand_mesh, face_mesh
                ]).export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_hand_face.obj")

            final_mesh = poisson(
                sum(full_lst),
                f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj",
                8,
            )

            dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
            rotate_recon_lst = dataset.render.get_image(cam_type="four")
            per_loop_lst.extend([data['image'][idx:idx + 1]] + rotate_recon_lst)

            # for video rendering
            in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
            in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

        # always export visualized png regardless of the cloth refinment

        per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))

        # visualize the final result
        per_data_lst[-1].save(os.path.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))

        os.makedirs(os.path.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
        in_tensor["uncrop_param"] = data["uncrop_param"]
        in_tensor["img_raw"] = data["img_raw"]
        torch.save(in_tensor, os.path.join(args.out_dir, cfg.name, "vid/in_tensor.pt"))

        # always export visualized video regardless of the cloth refinment
        if args.export_video:

            torch.cuda.empty_cache()

            # visualize the final results in self-rotation mode
            verts_lst = in_tensor["body_verts"] + in_tensor["BNI_verts"]
            faces_lst = in_tensor["body_faces"] + in_tensor["BNI_faces"]

            # self-rotated video
            dataset.render.load_meshes(verts_lst, faces_lst)
            dataset.render.get_rendered_video_multi(
                in_tensor,
                os.path.join(args.out_dir, cfg.name, f"vid/{data['name']}_cloth.mp4"),
            )

        # garment extraction from deepfashion images
        if not (args.seg_dir is None):
            if final_mesh is not None:
                recon_obj = final_mesh.copy()

            os.makedirs(os.path.join(args.out_dir, cfg.name, "clothes"), exist_ok=True)
            os.makedirs(
                os.path.join(args.out_dir, cfg.name, "clothes", "info"),
                exist_ok=True,
            )
            for seg in data["segmentations"]:
                # These matrices work for PyMaf, not sure about the other hps type
                K = np.array([
                    [1.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 1.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, -0.5000, 0.0000],
                    [-0.0000, -0.0000, 0.5000, 1.0000],
                ]).T

                R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

                t = np.array([[-0.0, -0.0, 100.0]])
                clothing_obj = extract_cloth(recon_obj, seg, K, R, t, smpl_obj)
                if clothing_obj is not None:
                    cloth_type = seg["type"].replace(" ", "_")
                    cloth_info = {
                        "betas": optimed_betas,
                        "body_pose": optimed_pose,
                        "global_orient": optimed_orient,
                        "pose2rot": False,
                        "clothing_type": cloth_type,
                    }

                    file_id = f"{data['name']}_{cloth_type}"
                    with open(
                            os.path.join(
                                args.out_dir,
                                cfg.name,
                                "clothes",
                                "info",
                                f"{file_id}_info.pkl",
                            ),
                            "wb",
                    ) as fp:
                        pickle.dump(cloth_info, fp)

                    clothing_obj.export(
                        os.path.join(args.out_dir, cfg.name, "clothes", f"{file_id}.obj"))
                else:
                    print(
                        f"Unable to extract clothing of type {seg['type']} from image {data['name']}"
                    )
