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
from PIL import Image
import numpy as np
import argparse
from termcolor import colored
from pytorch3d.structures import Meshes
from tqdm.auto import tqdm
from apps.ICON import ICON
from lib.common.cloth_extraction import extract_cloth
from lib.common.config import cfg
from lib.common.render import image2vid
from lib.common.train_util import get_visibility, init_loss, rot6d_to_rotmat
from lib.net.local_affine import LocalAffine
from lib.renderer.mesh import compute_normal_batch
from lib.common.BNI import BNI
from lib.dataset.TestDataset import TestDataset
from lib.dataset.mesh_util import (
    load_checkpoint,
    save_normal_tensor,
    update_mesh_shape_prior_losses,
    get_optim_grid_image,
    blend_rgb_norm,
    remesh,
    possion,
    tensor2variable,
    SMPLX,
)
import pickle
import os

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
    parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=100)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pymaf")
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-BNI", action="store_true")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg",
                        "--config",
                        type=str,
                        default="./configs/icon-filter.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymaf/configs/pymaf_config.yaml")
    device = torch.device(f"cuda:{args.gpu_device}")

    cfg_show_list = [
        "test_gpus",
        [args.gpu_device],
        "mcube_res",
        256,
        "clean_mesh",
        True,
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    # load model and dataloader
    model = ICON(cfg)
    model = load_checkpoint(model, cfg)

    # SMPLX object
    SMPLX_object = SMPLX()

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "colab": args.colab,
        "has_det": True,  # w/ or w/o detection
        "hps_type": args.hps_type,  # pymaf/pare/pixie
    }

    if args.hps_type == "pixie" and "pamir" in args.config:
        print(
            colored("PIXIE isn't compatible with PaMIR, thus switch to PyMAF",
                    "red"))
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

        Image.fromarray(data["img_crop"].astype(np.uint8)).save(
            os.path.join(args.out_dir, cfg.name, "png",
                         f"{data['name']}_crop.png"))

        in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}

        # The optimizer and variables
        optimed_pose = data["body_pose"].requires_grad_(True)
        optimed_trans = data["trans"].requires_grad_(True)
        optimed_betas = data["betas"].requires_grad_(True)
        optimed_orient = data["global_orient"].requires_grad_(True)

        optimizer_smpl = torch.optim.Adam(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
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

        # smpl optimization
        loop_smpl = tqdm(
            range(args.loop_smpl if cfg.net.prior_type != "pifu" else 1))

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        for i in loop_smpl:

            per_loop_lst = []

            optimizer_smpl.zero_grad()

            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(
                -1, 6)).unsqueeze(0)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(
                -1, 6)).unsqueeze(0)

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
                    left_hand_pose=tensor2variable(data["left_hand_pose"],
                                                   device),
                    right_hand_pose=tensor2variable(data["right_hand_pose"],
                                                    device),
                )

                smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"]

            # landmark errors
            if data["type"] == "smpl":
                smpl_joints_3d = (smpl_joints[0, :, :] + 1.0) * 0.5
            elif data["type"] == "smplx" and dataset_param[
                    "hps_type"] != "pixie":
                smpl_joints_3d = (
                    smpl_joints[0, dataset.smpl_joint_ids_45, :] + 1.0) * 0.5
            else:
                smpl_joints_3d = (
                    smpl_joints[0, dataset.smpl_joint_ids_45_pixie, :] +
                    1.0) * 0.5

            ghum_lmks = torch.tensor(data["landmark_dict"]["pose_landmarks"])[
                SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
            ghum_conf = torch.tensor(data["landmark_dict"]["pose_landmarks"])[
                SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
            smpl_lmks = smpl_joints_3d[SMPLX_object.ghum_smpl_pairs[:, 1], :2]
            losses["joint"]["value"] = (
                torch.norm(ghum_lmks - smpl_lmks, dim=1) * ghum_conf).mean()

            # render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor[
                "T_normal_B"] = dataset.render_normal(
                    smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                    in_tensor["smpl_faces"],
                )

            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor[
                    "normal_B"] = model.netG.normal_filter(in_tensor)

            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] -
                                    in_tensor["normal_F"])
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] -
                                    in_tensor["normal_B"])

            losses["normal"]["value"] = (diff_F_smpl +
                                         diff_B_smpl).mean() / 2.0

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)[0]
            gt_arr = torch.cat(
                [in_tensor["normal_F"][0], in_tensor["normal_B"][0]],
                dim=2).permute(1, 2, 0)
            gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
            bg_color = (torch.Tensor([0.5, 0.5, 0.5
                                      ]).unsqueeze(0).unsqueeze(0).to(device))
            gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()

            # for loose clothing, reply more on landmarks
            diff_ratio = diff_S.sum() / diff_S.shape.numel()
            if diff_ratio > 0.15:
                losses["joint"]["weight"] = 5e1

            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = "Body Fitting --- "
            for k in ["normal", "silhouette", "joint"]:
                pbar_desc += f"{k}: {losses[k]['value'] * losses[k]['weight']:.3f} | "
                smpl_loss += losses[k]["value"] * losses[k]["weight"]
            pbar_desc += f"Total: {smpl_loss:.3f}"
            loop_smpl.set_description(pbar_desc)

            # save intermediate results / vis_freq and final_step

            if (i % args.vis_freq == 0) or (i == args.loop_smpl - 1):

                # render optimized mesh as vitruvian color [-1,1]
                with torch.no_grad():

                    dataset.render.colored_meshes(SMPLX().cmap_smpl_vids(
                        type="smpl" if smpl_verts.shape[1] ==
                        6890 else "smplx"))
                    (
                        in_tensor["T_vitruvian_F"],
                        in_tensor["T_vitruvian_B"],
                    ) = dataset.render.get_image()

                per_loop_lst.extend([
                    in_tensor["image"],
                    in_tensor["T_normal_F"],
                    in_tensor["normal_F"],
                    # in_tensor["T_vitruvian_F"],
                    # diff_F_smpl / 2.0,
                    diff_S[:, :512].unsqueeze(0).unsqueeze(0).repeat(
                        1, 3, 1, 1),
                ])
                per_loop_lst.extend([
                    in_tensor["image"],
                    in_tensor["T_normal_B"],
                    in_tensor["normal_B"],
                    # in_tensor["T_vitruvian_B"],
                    # diff_B_smpl / 2.0,
                    diff_S[:,
                           512:].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
                ])
                per_data_lst.append(
                    get_optim_grid_image(per_loop_lst,
                                         None,
                                         nrow=len(per_loop_lst) // 2,
                                         type="smpl"))

            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)
            in_tensor["smpl_verts"] = smpl_verts * torch.tensor(
                [1.0, 1.0, -1.0]).to(device)

        # visualize the optimization process
        # 1. SMPL Fitting

        if args.vis_freq == 1:

            image2vid(
                per_data_lst,
                os.path.join(args.out_dir, cfg.name,
                             f"refinement/{data['name']}_smpl.avi"),
            )

        per_data_lst[-1].save(
            os.path.join(args.out_dir, cfg.name,
                         f"png/{data['name']}_smpl.png"))

        rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
        rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)

        Image.fromarray(
            np.concatenate(
                [data["img_ori"].astype(np.uint8), rgb_norm_F, rgb_norm_B],
                axis=1)).save(
                    os.path.join(args.out_dir, cfg.name,
                                 f"png/{data['name']}_overlap.png"))

        smpl_obj = trimesh.Trimesh(
            in_tensor["smpl_verts"].detach().cpu()[0] *
            torch.tensor([1.0, -1.0, 1.0]),
            in_tensor["smpl_faces"].detach().cpu()[0],
            process=False,
            maintains_order=True,
        )
        smpl_obj.export(
            f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl.obj")

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

        # ------------------------------------------------------------------------------------------------------------------
        # clothing refinement

        if True:

            per_data_lst = []

            # cloth recon
            in_tensor.update(
                dataset.compute_vis_cmap(in_tensor["smpl_verts"][0],
                                         in_tensor["smpl_faces"][0]))

            in_tensor.update({
                "smpl_norm":
                compute_normal_batch(in_tensor["smpl_verts"],
                                     in_tensor["smpl_faces"])
            })

            if cfg.net.prior_type == "pamir":
                in_tensor.update(
                    dataset.compute_voxel_verts(
                        optimed_pose,
                        optimed_orient,
                        optimed_betas,
                        optimed_trans,
                        data["scale"],
                    ))

            with torch.no_grad():
                verts_pr, faces_pr, _ = model.test_single(in_tensor)

            # ICON reconstruction w/o any optimization

            recon_obj = trimesh.Trimesh(verts_pr,
                                        faces_pr,
                                        process=False,
                                        maintains_order=True)
            recon_obj.export(
                os.path.join(args.out_dir, cfg.name,
                             f"obj/{data['name']}_recon.obj"))

            # Isotropic Explicit Remeshing for better geometry topology
            verts_remesh, faces_remesh, remeshed_mesh = remesh(
                os.path.join(args.out_dir, cfg.name,
                             f"obj/{data['name']}_recon.obj"),
                0.5,
                device,
            )

            if args.BNI:

                # # replace ICON by SMPL to provide depth-prior and side surfaces
                # verts_remesh = in_tensor["smpl_verts"].detach() * torch.tensor([1.0, -1.0, 1.0], device=device)
                # faces_remesh = in_tensor["smpl_faces"].detach()

                # rendering depth map for BNI
                in_tensor["verts_pr"] = verts_remesh
                in_tensor["faces_pr"] = faces_remesh
                in_tensor["depth_F"], in_tensor[
                    "depth_B"] = dataset.render_depth(verts_remesh,
                                                      faces_remesh)

                # save normals, depths and masks
                BNI_dict = save_normal_tensor(
                    in_tensor,
                    os.path.join(args.out_dir, cfg.name,
                                 f"BNI/{data['name']}"),
                )

                # BNI process
                BNI_object = BNI(
                    dir_path=os.path.join(args.out_dir, cfg.name, "BNI"),
                    name=data["name"],
                    in_tensor=in_tensor,
                    device=device,
                )

                BNI_object.extract_surface()

                # Possion Fusion: invisible ICON regions + visible BNI surfaces

                side_mesh = remeshed_mesh.copy()
                (xy, z) = verts_remesh.squeeze(0).split([2, 1], dim=1)
                F_vis = get_visibility(xy, z,
                                       faces_remesh.squeeze(0)[:, [0, 2, 1]])
                B_vis = get_visibility(xy, -z, faces_remesh.squeeze(0))
                verts_mask = (1.0 - F_vis[:, 0]) * (1.0 - B_vis[:, 0])
                faces_mask = verts_mask[faces_remesh.squeeze(0)].any(dim=1)
                side_mesh.update_faces(faces_mask)
                side_mesh.remove_unreferenced_vertices()

                side_mesh.export(
                    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_side.obj")

                possion_mesh = trimesh.Trimesh(
                    np.concatenate(
                        [side_mesh.vertices, BNI_object.F_B_trimesh.vertices],
                        axis=0),
                    np.concatenate(
                        [
                            side_mesh.faces,
                            BNI_object.F_B_trimesh.faces +
                            len(side_mesh.vertices),
                        ],
                        axis=0,
                    ),
                )

                final_mesh = possion(
                    possion_mesh,
                    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_combine.obj",
                )

                dataset.render.load_meshes(final_mesh.vertices,
                                           final_mesh.faces)
                rotate_recon_lst = dataset.render.get_image(
                    cam_ids=[0, 1, 2, 3])
                per_loop_lst.extend(rotate_recon_lst)

            else:

                # define local_affine deform verts
                mesh_pr = Meshes(verts_remesh, faces_remesh).to(device)
                local_affine_model = LocalAffine(
                    mesh_pr.verts_padded().shape[1],
                    mesh_pr.verts_padded().shape[0],
                    mesh_pr.edges_packed(),
                ).to(device)
                optimizer_cloth = torch.optim.Adam(
                    [{
                        "params": local_affine_model.parameters()
                    }],
                    lr=1e-4,
                    amsgrad=True)

                scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_cloth,
                    mode="min",
                    factor=0.1,
                    verbose=0,
                    min_lr=1e-5,
                    patience=args.patience,
                )

                if args.loop_cloth > 0:

                    loop_cloth = tqdm(range(args.loop_cloth))

                    for i in loop_cloth:

                        per_loop_lst = []

                        optimizer_cloth.zero_grad()

                        deformed_verts, stiffness, rigid = local_affine_model(
                            verts_remesh.to(device), return_stiff=True)
                        mesh_pr = mesh_pr.update_padded(deformed_verts)

                        # losses for laplacian, edge, normal consistency
                        update_mesh_shape_prior_losses(mesh_pr, losses)

                        (
                            in_tensor["P_normal_F"],
                            in_tensor["P_normal_B"],
                        ) = dataset.render_normal(mesh_pr.verts_padded(),
                                                  mesh_pr.faces_padded())

                        diff_F_cloth = torch.abs(in_tensor["P_normal_F"] -
                                                 in_tensor["normal_F"])
                        diff_B_cloth = torch.abs(in_tensor["P_normal_B"] -
                                                 in_tensor["normal_B"])

                        losses["cloth"]["value"] = (diff_F_cloth +
                                                    diff_B_cloth).mean()
                        losses["stiffness"]["value"] = torch.mean(stiffness)
                        losses["rigid"]["value"] = torch.mean(rigid)

                        # Weighted sum of the losses
                        cloth_loss = torch.tensor(
                            0.0, requires_grad=True).to(device)
                        pbar_desc = "Cloth Refinement --- "

                        for k in losses.keys():
                            if (k not in ["normal", "silhouette", "joint"]
                                    and losses[k]["weight"] > 0.0):
                                cloth_loss = (
                                    cloth_loss +
                                    losses[k]["value"] * losses[k]["weight"])
                                pbar_desc += f"{k}:{losses[k]['value']* losses[k]['weight']:.5f} | "

                        pbar_desc += f"Total: {cloth_loss:.5f}"
                        loop_cloth.set_description(pbar_desc)

                        # update params
                        cloth_loss.backward(retain_graph=True)
                        optimizer_cloth.step()
                        scheduler_cloth.step(cloth_loss)

                        # for vis
                        with torch.no_grad():

                            if (i % args.vis_freq) == 0 or (
                                    i == args.loop_cloth - 1):

                                rotate_recon_lst = dataset.render.get_image(
                                    cam_ids=[0, 1, 2, 3])

                                per_loop_lst.extend([
                                    in_tensor["image"],
                                    in_tensor["P_normal_F"],
                                    in_tensor["normal_F"],
                                    diff_F_cloth / 2.0,
                                ])
                                per_loop_lst.extend([
                                    in_tensor["image"],
                                    in_tensor["P_normal_B"],
                                    in_tensor["normal_B"],
                                    diff_B_cloth / 2.0,
                                ])
                                per_loop_lst.extend(rotate_recon_lst)
                                per_data_lst.append(
                                    get_optim_grid_image(
                                        per_loop_lst,
                                        None,
                                        nrow=len(per_loop_lst) // 2,
                                        type="cloth"))

                    # visualize the optimization process
                    # 2. Clothes Refinement

                    if args.vis_freq == 1:
                        image2vid(
                            per_data_lst,
                            os.path.join(
                                args.out_dir,
                                cfg.name,
                                f"refinement/{data['name']}_cloth.avi",
                            ),
                        )

                final_mesh = trimesh.Trimesh(
                    mesh_pr.verts_packed().detach().squeeze(0).cpu(),
                    mesh_pr.faces_packed().detach().squeeze(0).cpu(),
                    process=False,
                    maintains_order=True,
                )

                final_mesh.export(
                    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_refine.obj")

            # always export visualized png regardless of the cloth refinment

            per_data_lst.append(
                get_optim_grid_image(per_loop_lst, None, nrow=4, type="cloth"))

            # visualize the final result
            per_data_lst[-1].save(
                os.path.join(args.out_dir, cfg.name,
                             f"png/{data['name']}_cloth.png"))

            # always export visualized video regardless of the cloth refinment
            if args.export_video:

                # visualize the final results in self-rotation mode
                os.makedirs(os.path.join(args.out_dir, cfg.name, "vid"),
                            exist_ok=True)

                verts_lst = [smpl_obj.vertices, final_mesh.vertices]
                faces_lst = [smpl_obj.faces, final_mesh.faces]

                # self-rotated video
                dataset.render.load_meshes(verts_lst, faces_lst)
                dataset.render.get_rendered_video(
                    [data["img_ori"], rgb_norm_F, rgb_norm_B],
                    os.path.join(args.out_dir, cfg.name,
                                 f"vid/{data['name']}_cloth.mp4"),
                )

            # garment extraction from deepfashion images
            if not (args.seg_dir is None):
                if final_mesh is not None:
                    recon_obj = final_mesh.copy()

                os.makedirs(os.path.join(args.out_dir, cfg.name, "clothes"),
                            exist_ok=True)
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

                    R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                  [0.0, 0.0, -1.0]])

                    t = np.array([[-0.0, -0.0, 100.0]])
                    clothing_obj = extract_cloth(recon_obj, seg, K, R, t,
                                                 smpl_obj)
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
                            os.path.join(args.out_dir, cfg.name, "clothes",
                                         f"{file_id}.obj"))
                    else:
                        print(
                            f"Unable to extract clothing of type {seg['type']} from image {data['name']}"
                        )
