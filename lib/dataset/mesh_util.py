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

import json
import os
import os.path as osp

import _pickle as cPickle
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree

import lib.smplx as smplx
from lib.common.render_utils import Pytorch3dRasterizer, face_vertices


class Format:
    end = '\033[0m'
    start = '\033[4m'


class SMPLX:
    def __init__(self):

        self.current_dir = osp.join(osp.dirname(__file__), "../../data/smpl_related")

        self.smpl_verts_path = osp.join(self.current_dir, "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir, "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir, "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir, "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir, "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir, "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid_path = osp.join(self.current_dir, "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid_path = osp.join(self.current_dir, "smpl_data/fill_mouth_fid.npy")
        self.smplx_flame_vid_path = osp.join(
            self.current_dir, "smpl_data/FLAME_SMPLX_vertex_ids.npy"
        )
        # smpl & smpl-x vertex semantic labels
        self.smpl_vert_seg_path = osp.join(
            osp.dirname(__file__), "../../lib/common/smpl_vert_segmentation.json"
        )
        self.smplx_vert_seg_path = osp.join(
            osp.dirname(__file__), "../../lib/common/smplx_vert_segmentation.json"
        )

        self.front_flame_path = osp.join(self.current_dir, "smpl_data/FLAME_face_mask_ids.npy")
        self.smplx_vertex_lmkid_path = osp.join(
            self.current_dir, "smpl_data/smplx_vertex_lmkid.npy"
        )

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)
        self.smplx_vertex_lmkid = np.load(self.smplx_vertex_lmkid_path)

        self.smpl_vert_seg = json.load(open(self.smpl_vert_seg_path))
        self.smplx_vert_seg = json.load(open(self.smplx_vert_seg_path))

        # hand vertex ids
        self.smpl_mano_vid = np.unique(
            np.concatenate([
                self.smpl_vert_seg["rightHand"],
                self.smpl_vert_seg["rightHandIndex1"],
                self.smpl_vert_seg["leftHand"],
                self.smpl_vert_seg["leftHandIndex1"],
            ])
        )

        self.smplx_mano_vid = np.unique(
            np.concatenate([
                self.smplx_vert_seg["rightHand"],
                self.smplx_vert_seg["rightHandIndex1"],
                self.smplx_vert_seg["leftHand"],
                self.smplx_vert_seg["leftHandIndex1"],
            ])
        )

        # leg vertex ids

        self.smplx_leg_vid = np.unique(
            np.concatenate([
                self.smplx_vert_seg["rightUpLeg"],
                self.smplx_vert_seg["leftUpLeg"],
            ])
        )

        # eyeball and mouth face ids
        self.smplx_eyeball_fid_mask = np.load(self.smplx_eyeball_fid_path)
        self.smplx_eyeball_vid = self.smplx_faces[self.smplx_eyeball_fid_mask].flatten()
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid_path)

        self.smplx_flame_vid = np.load(self.smplx_flame_vid_path, allow_pickle=True)
        self.smplx_front_flame_vid = self.smplx_flame_vid[np.load(self.front_flame_path)]

        # hands vertex mask
        self.smplx_mano_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_mano_vid, dtype=torch.int64), 1.0
        )
        self.smpl_mano_vertex_mask = torch.zeros(self.smpl_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smpl_mano_vid, dtype=torch.int64), 1.0
        )

        # face vertex mask
        self.front_flame_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_front_flame_vid, dtype=torch.int64), 1.0
        )
        self.eyeball_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_eyeball_vid, dtype=torch.int64), 1.0
        )

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, "rb"))

        self.model_dir = osp.join(self.current_dir, "models")

        self.ghum_smpl_pairs = torch.tensor([(0, 24), (2, 26), (5, 25), (7, 28), (8, 27), (11, 16),
                                             (12, 17), (13, 18), (14, 19), (15, 20), (16, 21),
                                             (17, 39), (18, 44), (19, 36), (20, 41), (21, 35),
                                             (22, 40), (25, 4), (26, 5), (27, 7), (28, 8), (29, 31),
                                             (30, 34), (31, 29), (32, 32)]).long()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [61 + 68, 72 + 68]
        self.smpl_joint_ids_45 = np.arange(22).tolist() + [68, 73] + np.arange(55, 76).tolist()

        self.extra_joint_ids = np.array([
            61, 72, 66, 69, 58, 68, 57, 56, 64, 59, 67, 75, 70, 65, 60, 61, 63, 62, 76, 71, 72, 74,
            73
        ])

        self.extra_joint_ids += 68

        self.smpl_joint_ids_45_pixie = (np.arange(22).tolist() + self.extra_joint_ids.tolist())

    def cmap_smpl_vids(self, type):

        # smplx_to_smpl.pkl
        # KEYS:
        # closest_faces -   [6890, 3] with smplx vert_idx
        # bc            -   [6890, 3] with barycentric weights

        cmap_smplx = torch.as_tensor(np.load(self.cmap_vert_path)).float()

        if type == "smplx":
            return cmap_smplx

        elif type == "smpl":
            bc = torch.as_tensor(self.smplx_to_smpl["bc"].astype(np.float32))
            closest_faces = self.smplx_to_smpl["closest_faces"].astype(np.int32)
            cmap_smpl = torch.einsum("bij, bi->bj", cmap_smplx[closest_faces], bc)
            return cmap_smpl


model_init_params = dict(
    gender="male",
    model_type="smplx",
    model_path=SMPLX().model_dir,
    create_global_orient=False,
    create_body_pose=False,
    create_betas=False,
    create_left_hand_pose=False,
    create_right_hand_pose=False,
    create_expression=False,
    create_jaw_pose=False,
    create_leye_pose=False,
    create_reye_pose=False,
    create_transl=False,
    num_pca_comps=12,
)


def get_smpl_model(model_type, gender):
    return smplx.create(**model_init_params)


def load_fit_body(fitted_path, scale, smpl_type="smplx", smpl_gender="neutral", noise_dict=None):

    param = np.load(fitted_path, allow_pickle=True)
    for key in param.keys():
        param[key] = torch.as_tensor(param[key])

    smpl_model = get_smpl_model(smpl_type, smpl_gender)
    model_forward_params = dict(
        betas=param["betas"],
        global_orient=param["global_orient"],
        body_pose=param["body_pose"],
        left_hand_pose=param["left_hand_pose"],
        right_hand_pose=param["right_hand_pose"],
        jaw_pose=param["jaw_pose"],
        leye_pose=param["leye_pose"],
        reye_pose=param["reye_pose"],
        expression=param["expression"],
        return_verts=True,
    )

    if noise_dict is not None:
        model_forward_params.update(noise_dict)

    smpl_out = smpl_model(**model_forward_params)

    smpl_verts = ((smpl_out.vertices[0] * param["scale"] + param["translation"]) * scale).detach()
    smpl_joints = ((smpl_out.joints[0] * param["scale"] + param["translation"]) * scale).detach()
    smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False, maintain_order=True)

    return smpl_mesh, smpl_joints


def apply_face_mask(mesh, face_mask):

    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def apply_vertex_mask(mesh, vertex_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)

    return mesh


def apply_vertex_face_mask(mesh, vertex_mask, face_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1) * torch.tensor(face_mask)
    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def part_removal(full_mesh, part_mesh, thres, device, smpl_obj, region, clean=True):

    smpl_tree = cKDTree(smpl_obj.vertices)
    SMPL_container = SMPLX()

    from lib.dataset.PointFeat import PointFeat

    part_extractor = PointFeat(
        torch.tensor(part_mesh.vertices).unsqueeze(0).to(device),
        torch.tensor(part_mesh.faces).unsqueeze(0).to(device)
    )

    (part_dist, _) = part_extractor.query(torch.tensor(full_mesh.vertices).unsqueeze(0).to(device))

    remove_mask = part_dist < thres

    if region == "hand":
        _, idx = smpl_tree.query(full_mesh.vertices, k=1)
        if smpl_obj.vertices.shape[0] > 6890:
            full_lmkid = SMPL_container.smplx_vertex_lmkid[idx]
            remove_mask = torch.logical_and(
                remove_mask,
                torch.tensor(full_lmkid >= 20).type_as(remove_mask).unsqueeze(0)
            )
        else:
            remove_mask = torch.logical_and(
                remove_mask,
                torch.isin(
                    torch.tensor(idx).long(),
                    torch.tensor(SMPL_container.smpl_mano_vid).long()
                ).type_as(remove_mask).unsqueeze(0)
            )

    elif region == "face":
        _, idx = smpl_tree.query(full_mesh.vertices, k=5)
        face_space_mask = torch.isin(
            torch.tensor(idx), torch.tensor(SMPL_container.smplx_front_flame_vid)
        )
        remove_mask = torch.logical_and(
            remove_mask,
            face_space_mask.any(dim=1).type_as(remove_mask).unsqueeze(0)
        )

    BNI_part_mask = ~(remove_mask).flatten()[full_mesh.faces].any(dim=1)
    full_mesh.update_faces(BNI_part_mask.detach().cpu())
    full_mesh.remove_unreferenced_vertices()

    if clean:
        full_mesh = clean_floats(full_mesh)

    return full_mesh


class HoppeMesh:
    def __init__(self, verts, faces, uvs=None, texture=None):
        """
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        """

        # self.device = torch.device("cuda:0")
        mesh = trimesh.Trimesh(verts, faces, process=False, maintains_order=True)
        self.verts = torch.tensor(verts).float()
        self.faces = torch.tensor(faces).long()
        self.vert_normals = torch.tensor(mesh.vertex_normals).float()

        if (uvs is not None) and (texture is not None):
            self.vertex_colors = trimesh.visual.color.uv_to_color(uvs, texture)
            self.face_normals = torch.tensor(mesh.face_normals).float()

    def get_colors(self, points, faces):
        """
        Get colors of surface points from texture image through 
        barycentric interpolation.
        - points: [n, 3]
        - return: [n, 4] rgba
        """
        triangles = self.verts[faces]    #[n, 3, 3]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, points)    #[n, 3]
        vert_colors = self.vertex_colors[faces]    #[n, 3, 4]
        point_colors = torch.tensor((barycentric[:, :, None] * vert_colors).sum(axis=1)).float()
        return point_colors

    def triangles(self):
        return self.verts[self.faces].numpy()    #[n, 3, 3]


def tensor2variable(tensor, device):
    return tensor.requires_grad_(True).to(device)


def mesh_edge_loss(meshes, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor([0.0], dtype=torch.float32, device=meshes.device, requires_grad=True)

    N = len(meshes)
    edges_packed = meshes.edges_packed()    # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()    # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()    # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()    # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length)**2.0
    loss_vertex = loss * weights
    # loss_outlier = torch.topk(loss, 100)[0].mean()
    # loss_all = (loss_vertex.sum() + loss_outlier.mean()) / N
    loss_all = loss_vertex.sum() / N

    return loss_all


def remesh_laplacian(mesh, obj_path, face_count=50000):

    if mesh.faces.shape[0] != face_count:
        mesh = mesh.simplify_quadratic_decimation(face_count)
    mesh = trimesh.smoothing.filter_humphrey(
        mesh, alpha=0.1, beta=0.5, iterations=10, laplacian_operator=None
    )
    mesh.export(obj_path)

    return mesh


def poisson(mesh, obj_path, depth=10, face_count=50000, laplacian_remeshing=False):

    pcd_path = obj_path[:-4] + "_soups.ply"
    assert (mesh.vertex_normals.shape[1] == 3)
    mesh.export(pcd_path)
    pcl = o3d.io.read_point_cloud(pcd_path)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=depth, n_threads=-1
        )

    # only keep the largest component
    largest_mesh = keep_largest(trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles)))

    # mesh decimation for faster rendering
    low_res_mesh = largest_mesh.simplify_quadratic_decimation(face_count)
    if laplacian_remeshing:
        low_res_mesh = trimesh.smoothing.filter_humphrey(
            low_res_mesh, alpha=0.1, beta=0.5, iterations=10, laplacian_operator=None
        )
    low_res_mesh.export(obj_path)

    return low_res_mesh


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):

    # and (b) the edge length of the predicted mesh
    losses["edge"]["value"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["nc"]["value"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["lapla"]["value"] = mesh_laplacian_smoothing(mesh, method="uniform")


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, "vertices.txt"))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, "faces.txt"), dtype=np.int32) - 1
    smpl_face_code = (
        smpl_vertex_code[smpl_faces[:, 0]] + smpl_vertex_code[smpl_faces[:, 1]] +
        smpl_vertex_code[smpl_faces[:, 2]]
    ) / 3.0
    smpl_tetras = (np.loadtxt(os.path.join(folder, "tetrahedrons.txt"), dtype=np.int32) - 1)

    return_dict = {
        "smpl_vertex_code": torch.tensor(smpl_vertex_code), "smpl_face_code":
        torch.tensor(smpl_face_code), "smpl_faces": torch.tensor(smpl_faces), "smpl_tetras":
        torch.tensor(smpl_tetras)
    }

    return return_dict


def get_visibility(xy, z, faces, img_res=2**12, blur_radius=0.0, faces_per_pixel=1):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [B, N,2]
        z (torch.tensor): [B, N,1]
        faces (torch.tensor): [B, N,3]
        size (int): resolution of rendered image
    """

    if xy.ndimension() == 2:
        xy = xy.unsqueeze(0)
        z = z.unsqueeze(0)
        faces = faces.unsqueeze(0)

    xyz = (torch.cat((xy, -z), dim=-1) + 1.) / 2.
    N_body = xyz.shape[0]
    faces = faces.long().repeat(N_body, 1, 1)
    vis_mask = torch.zeros(size=(N_body, z.shape[1]))
    rasterizer = Pytorch3dRasterizer(image_size=img_res)

    meshes_screen = Meshes(verts=xyz, faces=faces)
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=rasterizer.raster_settings.image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=rasterizer.raster_settings.bin_size,
        max_faces_per_bin=rasterizer.raster_settings.max_faces_per_bin,
        perspective_correct=rasterizer.raster_settings.perspective_correct,
        cull_backfaces=rasterizer.raster_settings.cull_backfaces,
    )

    pix_to_face = pix_to_face.detach().cpu().view(N_body, -1)
    faces = faces.detach().cpu()

    for idx in range(N_body):
        Num_faces = len(faces[idx])
        vis_vertices_id = torch.unique(
            faces[idx][torch.unique(pix_to_face[idx][pix_to_face[idx] != -1]) - Num_faces * idx, :]
        )
        vis_mask[idx, vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def barycentric_coordinates_of_projection(points, vertices):
    """https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py"""
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    sb = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    sb[sb == 0] = 1e-6
    oneOver4ASquared = 1.0 / sb
    w = points - v0
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    vert_norms = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    face_norms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(face_norms)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vert_norms[faces[:, 0]] += face_norms
    vert_norms[faces[:, 1]] += face_norms
    vert_norms[faces[:, 2]] += face_norms
    normalize_v3(vert_norms)

    return vert_norms, face_norms


def compute_normal_batch(vertices, faces):

    if faces.shape[0] != vertices.shape[0]:
        faces = faces.repeat(vertices.shape[0], 1, 1)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    vert_norm = torch.zeros(bs * nv, 3).type_as(vertices)
    tris = face_vertices(vertices, faces)
    face_norm = F.normalize(
        torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]),
        dim=-1,
    )

    faces = (faces + (torch.arange(bs).type_as(faces) * nv)[:, None, None]).view(-1, 3)

    vert_norm[faces[:, 0]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 1]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 2]] += face_norm.view(-1, 3)

    vert_norm = F.normalize(vert_norm, dim=-1).view(bs, nv, 3)

    return vert_norm


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type="smpl"):

    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0), nrow=nrow, padding=0)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 * 255.0).astype(np.uint8)
    )

    if False:
        # add text
        draw = ImageDraw.Draw(grid_img)
        grid_size = 512
        if loss is not None:
            draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

        if type == "smpl":
            for col_id, col_txt in enumerate([
                "image",
                "smpl-norm(render)",
                "cloth-norm(pred)",
                "diff-norm",
                "diff-mask",
            ]):
                draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
        elif type == "cloth":
            for col_id, col_txt in enumerate([
                "image", "cloth-norm(recon)", "cloth-norm(pred)", "diff-norm"
            ]):
                draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
            for col_id, col_txt in enumerate(["0", "90", "180", "270"]):
                draw.text(
                    (10 + (col_id * grid_size), grid_size * 2 + 5),
                    col_txt,
                    (255, 0, 0),
                    font=font,
                )
        else:
            print(f"{type} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]), Image.ANTIALIAS)

    return grid_img


def clean_mesh(verts, faces):

    device = verts.device

    mesh_lst = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
    largest_mesh = keep_largest(mesh_lst)
    final_verts = torch.as_tensor(largest_mesh.vertices).float().to(device)
    final_faces = torch.as_tensor(largest_mesh.faces).long().to(device)

    return final_verts, final_faces


def clean_floats(mesh):
    thres = mesh.vertices.shape[0] * 1e-2
    mesh_lst = mesh.split(only_watertight=False)
    clean_mesh_lst = [mesh for mesh in mesh_lst if mesh.vertices.shape[0] > thres]
    return sum(clean_mesh_lst)


def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def mesh_move(mesh_lst, step, scale=1.0):

    trans = np.array([1.0, 0.0, 0.0]) * step

    resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=trans)

    results = []

    for mesh in mesh_lst:
        mesh.apply_transform(resize_matrix)
        results.append(mesh)

    return results


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0)):

    fitted_body = trimesh.load(fitted_path, process=False, maintain_order=True, skip_materials=True)
    resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    return np.array(fitted_body.vertices)


def get_joint_mesh(joints, radius=2.0):

    ball = trimesh.creation.icosphere(radius=radius)
    combined = None
    for joint in joints:
        ball_new = trimesh.Trimesh(vertices=ball.vertices + joint, faces=ball.faces, process=False)
        if combined is None:
            combined = ball_new
        else:
            combined = sum([combined, ball_new])
    return combined


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return (pcd_down, pcd_fpfh)


def o3d_ransac(src, dst):

    voxel_size = 0.01
    distance_threshold = 1.5 * voxel_size

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # print('Downsampling inputs')
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    # print('Running RANSAC')
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    )

    return result.transformation


def export_obj(v_np, f_np, vt, ft, path):

    # write mtl info into obj
    new_line = f"mtllib material.mtl \n"
    vt_lines = "\nusemtl mat0 \n"
    v_lines = ""
    f_lines = ""

    for _v in v_np:
        v_lines += f"v {_v[0]} {_v[1]} {_v[2]}\n"
    for fid, _f in enumerate(f_np):
        f_lines += f"f {_f[0]+1}/{ft[fid][0]+1} {_f[1]+1}/{ft[fid][1]+1} {_f[2]+1}/{ft[fid][2]+1}\n"
    for _vt in vt:
        vt_lines += f"vt {_vt[0]} {_vt[1]}\n"
    new_file_data = new_line + v_lines + vt_lines + f_lines
    with open(path, 'w') as file:
        file.write(new_file_data)
