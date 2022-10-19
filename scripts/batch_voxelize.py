import multiprocessing as mp
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import os.path as osp
from PIL import Image
import numpy as np
import torch, os
from lib.dataset.mesh_util import obj_loader, HoppeMesh, create_grid_points_from_xyz_bounds, projection
from scipy.spatial import cKDTree as KDTree   

types =  ["3dpeople","axyz","renderpeople","renderpeople_p27","humanalloy", "thuman2"]
scales = [1.0, 100.0, 1.0, 1.0, 2.54000508001016, 100.0] 

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

def grid_sampler(mesh_path):
    
    grid_pts = create_grid_points_from_xyz_bounds((-1, 1) * 3, 128).view(-1, 3).numpy()

    dataset = mesh_path.split("/")[-4]
    subject = mesh_path.split("/")[-2]
    scale = scales[types.index(dataset)]
    
    grid_path = f"./data/{dataset}/color_voxels/{subject}.npz"
    texture_path = osp.join("/".join(mesh_path.split("/")[:-1]), "material0.jpeg")
    calib_path = f"./data/{dataset}_36views/{subject}/calib/000.txt"
    
    if not osp.exists(grid_path):
        
        print(grid_path)

        mesh = load_textured_mesh(mesh_path, scale, texture_path)
        mesh_verts = projection(mesh.verts.numpy(), load_calib(calib_path))
        mesh_verts[:, 1] *= -1
        kdtree = KDTree(mesh_verts)
        _, grid_idx = kdtree.query(grid_pts)
        grid_colors = mesh.vertex_colors[grid_idx, :3] / 255.

        os.makedirs(osp.dirname(grid_path), exist_ok=True)
        np.savez_compressed(grid_path, grid_pts_rgb=grid_colors)
        
        print(f"Finish {grid_path}")


if __name__ == '__main__':
    
    paths = glob(f"./data/thuman2/scans/*/*.obj")

    print('Start voxelization.')
    
    p = Pool(mp.cpu_count())
    for _ in tqdm(p.imap_unordered(grid_sampler, paths), total=len(paths)):
        pass
    p.close()
    p.join()
    
    print('end voxelization.')