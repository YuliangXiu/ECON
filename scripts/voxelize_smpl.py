from lib.dataset.mesh_util import *
from lib.renderer.mesh import load_fit_body
import argparse
import os
import time
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

mesh_file = os.path.join(f"./data/{dataset}/smplx/{subject}.obj")

# create_grid_points_from_xyz_bounds

rescale_fitted_body, _ = load_fit_body(fit_file,
                                       180.0,
                                       smpl_type="smplx",
                                       smpl_gender="male")

smpl_verts = torch.from_numpy(rescale_fitted_body.vertices).cuda().float()
smpl_faces = torch.from_numpy(rescale_fitted_body.faces).cuda().long()

for y in range(0, 360, 360 // rotation):

    calib_file = os.path.join(f"{save_folder}/{subject}/calib", f"{y:03d}.txt")
    voxel_file = os.path.join(f"{save_folder}/{subject}/voxel", f"{y:03d}.npy")

    os.makedirs(os.path.dirname(vis_file), exist_ok=True)

    if not os.path.exists(vis_file):

        calib = load_calib(calib_file).cuda()
        calib_verts = projection(smpl_verts, calib, format="tensor")
        (xy, z) = calib_verts.split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, z, smpl_faces)

        if args.mode == "debug":
            mesh = trimesh.Trimesh(smpl_verts.cpu().numpy(),
                                   smpl_faces.cpu().numpy(),
                                   process=False)
            mesh.visual.vertex_colors = torch.tile(smpl_vis, (1, 3)).numpy()
            mesh.export(vis_file.replace("pt", "obj"))

        torch.save(smpl_vis, vis_file)

done_jobs = len(glob.glob(f"{save_folder}/*/vis"))
all_jobs = len(os.listdir(f"./data/{dataset}/scans"))
print(
    f"Finish visibility computing {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs"
)
