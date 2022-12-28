import os
import trimesh
import numpy as np
from .utils.libmesh import check_mesh_contains


def get_occ_gt(
    in_path=None,
    vertices=None,
    faces=None,
    pts_num=1000,
    points_sigma=0.01,
    with_dp=False,
    points=None,
    extra_points=None
):
    if in_path is not None:
        mesh = trimesh.load(in_path, process=False)
        print(type(mesh.vertices), mesh.vertices.shape, mesh.faces.shape)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # print('get_occ_gt', type(mesh.vertices), mesh.vertices.shape, mesh.faces.shape)

    # points_size = 100000
    points_padding = 0.1
    # points_sigma = 0.01
    points_uniform_ratio = 0.5
    n_points_uniform = int(pts_num * points_uniform_ratio)
    n_points_surface = pts_num - n_points_uniform

    if points is None:
        points_scale = 2.0
        boxsize = points_scale + points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = boxsize * (points_uniform - 0.5)
        points_surface, index_surface = mesh.sample(n_points_surface, return_index=True)
        points_surface += points_sigma * np.random.randn(n_points_surface, 3)
        points = np.concatenate([points_uniform, points_surface], axis=0)

    if extra_points is not None:
        extra_points += points_sigma * np.random.randn(len(extra_points), 3)
        points = np.concatenate([points, extra_points], axis=0)

    occupancies = check_mesh_contains(mesh, points)

    index_surface = None

    # points = points.astype(dtype)

    # print('occupancies', occupancies.dtype, np.sum(occupancies), occupancies.shape)
    # occupancies = np.packbits(occupancies)
    # print('occupancies bit', occupancies.dtype, np.sum(occupancies), occupancies.shape)

    # print('occupancies', points.shape, occupancies.shape, occupancies.dtype, np.sum(occupancies), index_surface.shape)

    return_dict = {}
    return_dict['points'] = points
    return_dict['points.occ'] = occupancies
    return_dict['sf_sidx'] = index_surface

    # export_pointcloud(mesh, modelname, loc, scale, args)
    # export_points(mesh, modelname, loc, scale, args)
    return return_dict
