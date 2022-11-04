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
import yaml
import os

from termcolor import colored
from tqdm.auto import tqdm
from apps.Normal import Normal
from apps.IFGeo import IFGeo
from apps.ICON import ICON
from lib.common.config import cfg
from lib.common.train_util import init_loss, load_normal_networks, load_networks
from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.dataset.TestDataset import TestDataset
from lib.net.geometry import rot6d_to_rotmat
from lib.dataset.mesh_util import *
from lib.common.voxelize import VoxelGrid

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=20)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-BNI", action="store_true")
    parser.add_argument("-multi", action="store_false")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")

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
    
    # load model
    normal_model = Normal(cfg).to(device)
    load_normal_networks(normal_model, cfg.normal_path)
    normal_model.netG.eval()

    if args.BNI:

        # load IFGeo model
        ifnet_model = IFGeo(cfg).to(device)
        load_networks(ifnet_model, mlp_path=cfg.ifnet_path)
        ifnet_model.netG.eval()

    else:

        # load model and dataloader
        icon_model = ICON(cfg).to(device)
        icon_model = load_checkpoint(icon_model, cfg)
        icon_model.eval()

    # SMPLX object
    SMPLX_object = SMPLX()
    dropbox_dir = osp.join(args.out_dir.replace("results", "/home/yxiu/Dropbox/ECON"), cfg.name)
    os.makedirs(dropbox_dir, exist_ok=True)

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "use_seg": True,  # w/ or w/o segmentation
        "hps_type": cfg.bni.hps_type,  # pymafx/pixie
        "vol_res": cfg.vol_res,
        "single": args.multi,
    }

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:

        losses = init_loss()

        bni_path = osp.join(args.out_dir, cfg.name, "BNI", f"{data['name']}.yaml")
        os.makedirs(osp.dirname(bni_path), exist_ok=True)

        if osp.exists(bni_path):
            cfg.bni.merge_from_file(bni_path)
            if cfg.bni.finish:
                continue
        else:
            cfg.bni.merge_from_list(["finish", True])
            with open(bni_path, "w+") as file:
                _ = yaml.dump(dict(cfg.bni), file)

        pbar.set_description(f"{data['name']}")

        # final results rendered as image
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)

        os.makedirs(osp.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes
        # 1. SMPL mesh (xxx_smpl.obj)
        # 2. SMPL params (xxx_smpl.npy)
        # 3. clohted mesh (xxx_recon.obj)
        # 4. remeshed clothed mesh (xxx_remesh.obj)
        # 5. refined clothed mesh (xxx_refine.obj)
        # 6. BNI clothed mesh (xxx_combine.obj)

        os.makedirs(osp.join(args.out_dir, cfg.name, "obj"), exist_ok=True)

        img_crop_path = osp.join(args.out_dir, cfg.name, "png", f"{data['name']}_crop.png")
        torchvision.utils.save_image(data["img_crop"], img_crop_path)

        in_tensor = {
            "smpl_faces": data["smpl_faces"],
            "image": data["img_icon"].to(device),
            "mask": data["img_mask"].to(device)
        }

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

        # [result_loop_1, result_loop_2, ...]
        per_data_lst = []

        N_body, N_pose = optimed_pose.shape[:2]

        if osp.exists(osp.join(args.out_dir, "econ-if", f"png/{data['name']}_smpl.png")):

            smpl_verts_lst = []
            smpl_faces_lst = []
            for idx in range(N_body):

                smpl_obj = f"{args.out_dir}/econ-if/obj/{data['name']}_smpl_{idx:02d}.obj"
                smpl_mesh = trimesh.load(smpl_obj)
                smpl_verts = torch.tensor(smpl_mesh.vertices).to(device).float()
                smpl_faces = torch.tensor(smpl_mesh.faces).to(device).long()
                smpl_verts_lst.append(smpl_verts)
                smpl_faces_lst.append(smpl_faces)

            batch_smpl_verts = torch.stack(smpl_verts_lst)
            batch_smpl_faces = torch.stack(smpl_faces_lst)

            # render optimized mesh as normal [-1,1]
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                batch_smpl_verts,
                batch_smpl_faces,
            )

            in_tensor["smpl_verts"] = batch_smpl_verts * torch.tensor([1., -1., 1.]).to(device)
            in_tensor["smpl_faces"] = batch_smpl_faces[:, :, [0, 2, 1]]

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_model.netG(in_tensor)
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
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor(
                    [1.0, 1.0, -1.0]).to(device)

                # landmark errors
                if data["type"] == "smpl":
                    smpl_joints_3d = (smpl_joints[0, :, :] + 1.0) * 0.5
                    in_tensor["smpl_joint"] = smpl_joints[:, :24, :]
                else:
                    smpl_joints_3d = (smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] +
                                      1.0) * 0.5
                    in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_data.
                                                          smpl_joint_ids_24_pixie, :]

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
                    in_tensor["normal_F"], in_tensor["normal_B"] = normal_model.netG(in_tensor)

                diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
                diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

                losses["normal"]["value"] = (diff_F_smpl + diff_B_smpl).mean() / 2.0

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
                # for highly occluded body, reply only on high-confidence landmarks
                # no silhouette+normal loss

                # BUG: PyTorch3D silhouette renderer generates dilated mask
                bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
                smpl_arr_fake = torch.cat([
                    in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                    in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
                ],
                                          dim=-1)

                body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)).sum(
                    dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])

                body_overlap_flag = body_overlap < cfg.body_overlap_thres
                losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) *
                                            ghum_conf).mean(dim=1)

                # Weighted sum of the losses
                smpl_loss = 0.0
                pbar_desc = "Body Fitting --- "
                for k in ["normal", "silhouette", "joint"]:
                    per_loop_loss = (losses[k]["value"] *
                                     torch.tensor(losses[k]["weight"]).to(device)).mean()
                    pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                    smpl_loss += per_loop_loss
                pbar_desc += f"Total: {smpl_loss:.3f}"
                loose_str = ''.join([str(j) for j in cloth_overlap_flag.int().tolist()])
                occlude_str = ''.join([str(j) for j in body_overlap_flag.int().tolist()])
                pbar_desc += colored(f"| loose:{loose_str}, occluded:{occlude_str}", "yellow")
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

            per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png"))
            # per_data_lst[-1].save(osp.join(dropbox_dir, f"{data['name']}_smpl.png"))

            rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
            rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)

            img_overlap_path = osp.join(args.out_dir, cfg.name, f"png/{data['name']}_overlap.png")
            torchvision.utils.save_image(
                torch.Tensor([data["img_raw"], rgb_norm_F, rgb_norm_B]).permute(0, 3, 1, 2) / 255.,
                img_overlap_path)
            torchvision.utils.save_image(
                torch.Tensor([data["img_raw"], rgb_norm_F, rgb_norm_B]).permute(0, 3, 1, 2) / 255.,
                osp.join(dropbox_dir, f"{data['name']}_overlap.png"))

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

        if args.BNI:

            in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
                batch_smpl_verts, batch_smpl_faces)

            per_loop_lst = []

            in_tensor["BNI_verts"] = []
            in_tensor["BNI_faces"] = []
            in_tensor["body_verts"] = []
            in_tensor["body_faces"] = []

            for idx in range(N_body):

                side_mesh = smpl_obj_lst[idx].copy()
                face_mesh = smpl_obj_lst[idx].copy()
                hand_mesh = smpl_obj_lst[idx].copy()

                # save normals, depths and masks
                BNI_dict = save_normal_tensor(
                    in_tensor,
                    idx,
                    osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
                    cfg.bni.thickness,
                )

                # BNI process
                BNI_object = BNI(dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
                                 name=data["name"],
                                 BNI_dict=BNI_dict,
                                 cfg=cfg.bni,
                                 device=device)

                BNI_object.extract_surface(False)

                in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
                in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())

                # requires shape completion when low overlap
                # replace SMPL by completed mesh as side_mesh

                if cfg.bni.always_ifnet:

                    if not cfg.bni.always_ifnet:
                        print(
                            colored(f"Low overlap: {body_overlap[idx]:.2f}, shape completion\n",
                                    "green"))

                    side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

                    # mesh completion via IF-net
                    in_tensor.update(
                        dataset.depth_to_voxel({
                            "depth_F": BNI_object.F_depth.unsqueeze(0),
                            "depth_B": BNI_object.B_depth.unsqueeze(0)
                        }))

                    occupancies = VoxelGrid.from_mesh(side_mesh,
                                                      cfg.vol_res,
                                                      loc=[
                                                          0,
                                                      ] * 3,
                                                      scale=2.0).data.transpose(2, 1, 0)
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
                F_vis = get_visibility(xy,
                                       z,
                                       side_faces[..., [0, 2, 1]],
                                       img_res=2**9,
                                       faces_per_pixel=1)
                B_vis = get_visibility(xy, -z, side_faces, img_res=2**9, faces_per_pixel=1)

                full_lst = []

                if "face" in cfg.bni.use_smpl:
                    # only face
                    face_mesh = apply_vertex_mask(face_mesh, SMPLX_object.front_flame_vertex_mask)
                    face_mesh.vertices[:, 2] -= cfg.bni.thickness
                    # remove face neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(BNI_object.F_B_trimesh,
                                                          None,
                                                          face_mesh,
                                                          cfg.bni.face_thres,
                                                          device,
                                                          camera_ray=True)
                    full_lst += [face_mesh]

                if "hand" in cfg.bni.use_smpl:
                    # only hands
                    hand_mesh = apply_vertex_mask(hand_mesh, SMPLX_object.mano_vertex_mask)
                    # remove face neighbor triangles
                    BNI_object.F_B_trimesh = part_removal(BNI_object.F_B_trimesh, None, hand_mesh,
                                                          cfg.bni.hand_thres, device)
                    full_lst += [hand_mesh]

                full_lst += [BNI_object.F_B_trimesh]

                # initial side_mesh could be SMPLX or IF-net
                side_mesh = part_removal(side_mesh,
                                         torch.logical_or(F_vis, B_vis),
                                         sum([face_mesh, hand_mesh]),
                                         4e-2,
                                         device,
                                         clean=False)

                full_lst += [side_mesh]

                # export intermediate meshes
                BNI_object.F_B_trimesh.export(
                    f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_BNI.obj")

                side_mesh.export(f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_side.obj")

                if cfg.bni.use_poisson:
                    final_mesh = poisson(
                        sum(full_lst),
                        f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj",
                        cfg.bni.poisson_depth,
                    )
                else:
                    final_mesh = sum(full_lst)

                dataset.render.load_meshes(final_mesh.vertices, final_mesh.faces)
                rotate_recon_lst = dataset.render.get_image(cam_type="four")
                per_loop_lst.extend([in_tensor['image'][idx:idx + 1]] + rotate_recon_lst)

                # for video rendering
                in_tensor["BNI_verts"].append(torch.tensor(final_mesh.vertices).float())
                in_tensor["BNI_faces"].append(torch.tensor(final_mesh.faces).long())

        else:

            per_loop_lst = []

            for idx in range(N_body):

                # cloth recon
                in_tensor.update(
                    dataset.compute_vis_cmap(in_tensor["smpl_verts"][idx:idx + 1],
                                             in_tensor["smpl_faces"][idx:idx + 1]))

                in_tensor.update({
                    "smpl_norm":
                        compute_normal_batch(in_tensor["smpl_verts"][idx:idx + 1],
                                             in_tensor["smpl_faces"][idx:idx + 1])
                })

                if cfg.net.prior_type == "pamir":
                    in_tensor.update(
                        dataset.compute_voxel_verts(
                            optimed_pose[idx:idx + 1],
                            optimed_orient[idx:idx + 1],
                            optimed_betas[idx:idx + 1],
                            optimed_trans[idx:idx + 1],
                            data["scale"],
                        ))

                # BNI does not need IF output

                with torch.no_grad():
                    verts_pr, faces_pr, _ = icon_model.test_single(in_tensor)

                # ICON reconstruction w/o any optimization

                recon_obj = trimesh.Trimesh(verts_pr, faces_pr, process=False, maintains_order=True)
                recon_obj.export(
                    os.path.join(args.out_dir, cfg.name, f"obj/{data['name']}_recon.obj"))

                dataset.render.load_meshes(recon_obj.vertices, recon_obj.faces)
                rotate_recon_lst = dataset.render.get_image(cam_type="four")
                per_loop_lst.extend([in_tensor['image'][idx:idx + 1]] + rotate_recon_lst)

        # always export visualized png regardless of the cloth refinment

        per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="cloth"))

        # visualize the final result
        per_data_lst[-1].save(osp.join(args.out_dir, cfg.name, f"png/{data['name']}_cloth.png"))
        per_data_lst[-1].save(osp.join(dropbox_dir, f"{data['name']}_cloth.png"))

        os.makedirs(osp.join(args.out_dir, cfg.name, "vid"), exist_ok=True)
        in_tensor["uncrop_param"] = data["uncrop_param"]
        in_tensor["img_raw"] = data["img_raw"]
        torch.save(in_tensor, osp.join(args.out_dir, cfg.name, "vid/in_tensor.pt"))

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
                osp.join(args.out_dir, cfg.name, f"vid/{data['name']}_cloth.mp4"),
            )
