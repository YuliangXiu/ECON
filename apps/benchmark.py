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

import torch
from termcolor import colored
from tqdm.auto import tqdm

from apps.IFGeo import IFGeo
from apps.Normal import Normal
from lib.common.BNI import BNI
from lib.common.BNI_utils import save_normal_tensor
from lib.common.config import cfg
from lib.common.voxelize import VoxelGrid
from lib.dataset.EvalDataset import EvalDataset
from lib.dataset.Evaluator import Evaluator
from lib.dataset.mesh_util import *

torch.backends.cudnn.benchmark = True
speed_analysis = False

if __name__ == "__main__":

    if speed_analysis:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-ifnet", action="store_true")
    parser.add_argument("-cfg", "--config", type=str, default="./configs/econ.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    device = torch.device("cuda:0")

    cfg_test_list = [
        "dataset.rotation_num",
        3,
        "bni.use_smpl",
        ["hand"],
        "bni.use_ifnet",
        args.ifnet,
        "bni.cut_intersection",
        True,
    ]

    # # if w/ RenderPeople+CAPE
    # cfg_test_list += ["dataset.types", ["cape", "renderpeople"], "dataset.scales", [100.0, 1.0]]

    # if only w/ CAPE
    cfg_test_list += ["dataset.types", ["cape"], "dataset.scales", [100.0]]

    cfg.merge_from_list(cfg_test_list)
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

    dataset = EvalDataset(cfg=cfg, device=device)
    evaluator = Evaluator(device=device)
    export_dir = osp.join(cfg.results_path, cfg.name, "IF-Net+" if cfg.bni.use_ifnet else "SMPL-X")
    print(colored(f"Dataset Size: {len(dataset)}", "green"))

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

    pbar = tqdm(dataset)
    benchmark = {}

    for data in pbar:

        for key in data.keys():
            if torch.is_tensor(data[key]):
                data[key] = data[key].unsqueeze(0).to(device)

        is_smplx = True if 'smplx_path' in data.keys() else False

        # filenames and makedirs
        current_name = f"{data['dataset']}-{data['subject']}-{data['rotation']:03d}"
        current_dir = osp.join(export_dir, data['dataset'], data['subject'])
        os.makedirs(current_dir, exist_ok=True)
        final_path = osp.join(current_dir, f"{current_name}_final.obj")

        if not osp.exists(final_path):

            in_tensor = data.copy()

            batch_smpl_verts = in_tensor["smpl_verts"].detach()
            batch_smpl_verts *= torch.tensor([1.0, -1.0, 1.0]).to(device)
            batch_smpl_faces = in_tensor["smpl_faces"].detach()

            in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(
                batch_smpl_verts, batch_smpl_faces
            )

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

            smpl_mesh = trimesh.Trimesh(
                batch_smpl_verts.cpu().numpy()[0],
                batch_smpl_faces.cpu().numpy()[0]
            )

            side_mesh = smpl_mesh.copy()
            face_mesh = smpl_mesh.copy()
            hand_mesh = smpl_mesh.copy()
            smplx_mesh = smpl_mesh.copy()

            # save normals, depths and masks
            BNI_dict = save_normal_tensor(
                in_tensor,
                0,
                osp.join(current_dir, "BNI/param_dict"),
                cfg.bni.thickness if data['dataset'] == 'renderpeople' else 0.0,
            )

            # BNI process
            BNI_object = BNI(
                dir_path=osp.join(current_dir, "BNI"),
                name=current_name,
                BNI_dict=BNI_dict,
                cfg=cfg.bni,
                device=device
            )

            BNI_object.extract_surface(False)

            if is_smplx:
                side_mesh = apply_face_mask(side_mesh, ~SMPLX_object.smplx_eyeball_fid_mask)

            if cfg.bni.use_ifnet:

                # mesh completion via IF-net
                in_tensor.update(
                    dataset.depth_to_voxel({
                        "depth_F": BNI_object.F_depth.unsqueeze(0).to(device), "depth_B":
                        BNI_object.B_depth.unsqueeze(0).to(device)
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

                side_mesh_path = osp.join(current_dir, f"{current_name}_IF.obj")
                side_mesh = remesh_laplacian(trimesh.Trimesh(verts_IF, faces_IF), side_mesh_path)

            full_lst = []

            if "hand" in cfg.bni.use_smpl:

                # only hands
                if is_smplx:
                    hand_mesh = apply_vertex_mask(hand_mesh, SMPLX_object.smplx_mano_vertex_mask)
                else:
                    hand_mesh = apply_vertex_mask(hand_mesh, SMPLX_object.smpl_mano_vertex_mask)

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
                # hand_mesh.export(osp.join(current_dir, f"{current_name}_hands.obj"))
                full_lst += [hand_mesh]

            full_lst += [BNI_object.F_B_trimesh]

            # initial side_mesh could be SMPLX or IF-net
            side_mesh = part_removal(
                side_mesh, sum(full_lst), 2e-2, device, smplx_mesh, region="", clean=False
            )

            full_lst += [side_mesh]

            if cfg.bni.use_poisson:
                final_mesh = poisson(
                    sum(full_lst),
                    final_path,
                    cfg.bni.poisson_depth,
                )
            else:
                final_mesh = sum(full_lst)
                final_mesh.export(final_path)
        else:
            final_mesh = trimesh.load(final_path)

        # evaluation
        metric_path = osp.join(export_dir, "metric.npy")

        if osp.exists(metric_path):
            benchmark = np.load(metric_path, allow_pickle=True).item()

        if benchmark == {} or data["dataset"] not in benchmark.keys(
        ) or f"{data['subject']}-{data['rotation']}" not in benchmark[data["dataset"]]["subject"]:

            result_eval = {
                "verts_gt": data["verts"][0],
                "faces_gt": data["faces"][0],
                "verts_pr": final_mesh.vertices,
                "faces_pr": final_mesh.faces,
                "calib": data["calib"][0],
            }

            evaluator.set_mesh(result_eval, scale=False)
            chamfer, p2s = evaluator.calculate_chamfer_p2s(num_samples=1000)
            nc = evaluator.calculate_normal_consist(osp.join(current_dir, f"{current_name}_nc.png"))

            if data["dataset"] not in benchmark.keys():
                benchmark[data["dataset"]] = {
                    "chamfer": [chamfer.item()],
                    "p2s": [p2s.item()],
                    "nc": [nc.item()],
                    "subject": [f"{data['subject']}-{data['rotation']}"],
                    "total": 1,
                }
            else:
                benchmark[data["dataset"]]["chamfer"] += [chamfer.item()]
                benchmark[data["dataset"]]["p2s"] += [p2s.item()]
                benchmark[data["dataset"]]["nc"] += [nc.item()]
                benchmark[data["dataset"]]["subject"] += [f"{data['subject']}-{data['rotation']}"]
                benchmark[data["dataset"]]["total"] += 1

            np.save(metric_path, benchmark, allow_pickle=True)

        else:

            subject_idx = benchmark[data["dataset"]
                                   ]["subject"].index(f"{data['subject']}-{data['rotation']}")
            chamfer = torch.tensor(benchmark[data["dataset"]]["chamfer"][subject_idx])
            p2s = torch.tensor(benchmark[data["dataset"]]["p2s"][subject_idx])
            nc = torch.tensor(benchmark[data["dataset"]]["nc"][subject_idx])

        pbar.set_description(
            f"{current_name} | {chamfer.item():.3f} | {p2s.item():.3f} | {nc.item():.4f}"
        )

    for dataset in benchmark.keys():
        for metric in ["chamfer", "p2s", "nc"]:
            print(
                f"{dataset}-{metric}: {sum(benchmark[dataset][metric])/benchmark[dataset]['total']:.4f}"
            )

    if cfg.bni.use_ifnet:
        print(colored("Finish evaluating on ECON_IF", "green"))
    else:
        print(colored("Finish evaluating of ECON_EX", "green"))

    if speed_analysis:
        profiler.disable()
        profiler.dump_stats(osp.join(export_dir, "econ.stats"))
        stats = pstats.Stats(osp.join(export_dir, "econ.stats"))
        stats.sort_stats("cumtime").print_stats(10)
