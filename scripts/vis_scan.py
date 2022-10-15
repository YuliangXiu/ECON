from lib.dataset.mesh_util import projection, load_calib, get_visibility
import argparse
import os
import time
import numpy as np
import trimesh
import torch
import glob

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", type=str, help="subject name")
parser.add_argument("-o", "--out_dir", type=str, help="output dir")
parser.add_argument("-r", "--rotation", type=str, help="rotation num")
parser.add_argument("-m", "--mode", type=str, help="gen/debug")

args = parser.parse_args()

subject = args.subject
save_folder = args.out_dir
rotation = int(args.rotation)

dataset = save_folder.split("/")[-1].split("_")[0]

mesh_file = os.path.join(f"./data/{dataset}/scans/{subject}", f"{subject}.obj")

scan_mesh = trimesh.load(mesh_file, process=False, maintain_order=True)
smpl_verts = torch.from_numpy(scan_mesh.vertices).cuda().float() * 100.
smpl_faces = torch.from_numpy(scan_mesh.faces).cuda().long()

for y in range(0, 360, 360 // rotation):

    calib_file = os.path.join(f"{save_folder.replace('debug', 'data')}/{subject}/calib", f"{y:03d}.txt")
    vis_file = os.path.join(f"{save_folder}/{subject}/vis", f"{y:03d}_scan.pt")

    os.makedirs(os.path.dirname(vis_file), exist_ok=True)

    if not os.path.exists(vis_file):

        calib = load_calib(calib_file).cuda()
        calib_verts = projection(smpl_verts, calib)
        (xy, z) = calib_verts.split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, z, smpl_faces, 2**8, 2.0, 150).squeeze(0).unsqueeze(-1)

        if args.mode == "debug":
            mesh = trimesh.Trimesh(calib_verts.cpu().numpy()*np.array([1.0,-1.0,1.0]),
                                   smpl_faces.cpu().numpy(),
                                   process=False, maintain_order=True)
            mesh.visual.vertex_colors = torch.tile(smpl_vis, (1, 3)).numpy() * 255
            mesh.export(os.path.join(save_folder, f"{subject}_{y:03d}.obj"))
            break

        torch.save(smpl_vis, vis_file)

done_jobs = len(glob.glob(f"{save_folder}/*/vis/*_scan.pt"))
all_jobs = len(os.listdir(f"./data/{dataset}/scans")) * rotation
print(
    f"Finish visibility computing {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs"
)
