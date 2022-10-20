import os.path as osp
from PIL import Image
import numpy as np
import torch, os
from lib.dataset.mesh_util import obj_loader, HoppeMesh, create_grid_points_from_xyz_bounds, projection
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def load_textured_mesh(mesh_path, scale, texture_path):

    verts, faces, verts_uv, _ = obj_loader(mesh_path)
    texture = Image.open(texture_path)

    mesh = HoppeMesh(verts * scale, faces, verts_uv, texture)

    return mesh


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def grid_sampler(calib_path, dataset, scale):

    grid_pts = create_grid_points_from_xyz_bounds((-1, 1) * 3, 256).view(-1, 3).numpy()

    subject = calib_path.split("/")[-3]
    rid = calib_path.split("/")[-1].split(".")[0]

    mesh_path = f"./data/{dataset}/scans/{subject}/{subject}.obj"
    texture_path = osp.join("/".join(mesh_path.split("/")[:-1]), "material0.jpeg")
    grid_path = f"./data/{dataset}_36views/{subject}/tex_voxels/{rid}.npz"
    real_calib_path = osp.join("./data", calib_path)

    if osp.exists(real_calib_path):

        if not osp.exists(grid_path) or np.load(
                grid_path, allow_pickle=True)["grid_pts_rgb"].shape[0] != 256**3:
            mesh = load_textured_mesh(mesh_path, scale, texture_path)
            mesh_verts = projection(mesh.verts.numpy(), load_calib(real_calib_path))
            mesh_verts[:, 1] *= -1
            kdtree = cKDTree(mesh_verts)
            _, grid_idx = kdtree.query(grid_pts, workers=-1)
            grid_colors = mesh.vertex_colors[grid_idx, :3] / 255.

            os.makedirs(osp.dirname(grid_path), exist_ok=True)
            np.savez_compressed(grid_path, grid_pts_rgb=grid_colors)

            print(grid_path)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-calib", "--calib_path", type=str, default="")
    args = parser.parse_args()

    grid_sampler(args.calib_path, 'thuman2', 100.0)
