
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

from lib.dataset.mesh_util import projection
from lib.common.render import Render
import numpy as np
import torch
import os.path as osp
from torchvision.utils import make_grid
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.structures import Pointclouds
from PIL import Image


def point_mesh_distance(meshes, pcls):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = torch.sqrt(point_to_face) * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist


class Evaluator:

    def __init__(self, device):

        self.render = Render(size=512, device=device)
        self.device = device

    def set_mesh(self, result_dict):

        for k, v in result_dict.items():
            setattr(self, k, v)

        self.verts_pr -= self.recon_size / 2.0
        self.verts_pr /= self.recon_size / 2.0
        self.verts_gt = projection(self.verts_gt, self.calib)
        self.verts_gt[:, 1] *= -1

        self.src_mesh = self.render.VF2Mesh(self.verts_pr, self.faces_pr)
        self.tgt_mesh = self.render.VF2Mesh(self.verts_gt, self.faces_gt)

    def calculate_normal_consist(self, normal_path):

        self.render.meshes = self.src_mesh
        src_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3])
        self.render.meshes = self.tgt_mesh
        tgt_normal_imgs = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3])

        src_normal_arr = (
            make_grid(torch.cat(src_normal_imgs, dim=0), nrow=4)+1.0)*0.5  # [0,1]
        tgt_normal_arr = (
            make_grid(torch.cat(tgt_normal_imgs, dim=0), nrow=4)+1.0)*0.5  # [0,1]
        
        error = (((src_normal_arr - tgt_normal_arr)
                 ** 2).sum(dim=0).mean()) * 4.0

        normal_img = Image.fromarray((torch.cat([src_normal_arr, tgt_normal_arr], dim=1).permute(
            1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))
        normal_img.save(normal_path)

        return error

    def export_mesh(self, dir, name):

        IO().save_mesh(self.src_mesh, osp.join(dir, f"{name}_src.obj"))
        IO().save_mesh(self.tgt_mesh, osp.join(dir, f"{name}_tgt.obj"))
        
    def calculate_chamfer_p2s(self, num_samples=1000):
    
        tgt_points = Pointclouds(
            sample_points_from_meshes(self.tgt_mesh, num_samples))
        src_points = Pointclouds(
            sample_points_from_meshes(self.src_mesh, num_samples))
        p2s_dist = point_mesh_distance(self.src_mesh, tgt_points) * 100.0
        chamfer_dist = (point_mesh_distance(
            self.tgt_mesh, src_points) * 100.0 + p2s_dist) * 0.5
        
        return chamfer_dist, p2s_dist

    def calc_acc(self, output, target, thres=0.5, use_sdf=False):

        # # remove the surface points with thres
        # non_surf_ids = (target != thres)
        # output = output[non_surf_ids]
        # target = target[non_surf_ids]

        with torch.no_grad():
            output = output.masked_fill(output < thres, 0.0)
            output = output.masked_fill(output > thres, 1.0)

            if use_sdf:
                target = target.masked_fill(target < thres, 0.0)
                target = target.masked_fill(target > thres, 1.0)

            acc = output.eq(target).float().mean()

            # iou, precison, recall
            output = output > thres
            target = target > thres

            union = output | target
            inter = output & target

            _max = torch.tensor(1.0).to(output.device)

            union = max(union.sum().float(), _max)
            true_pos = max(inter.sum().float(), _max)
            vol_pred = max(output.sum().float(), _max)
            vol_gt = max(target.sum().float(), _max)

            return acc, true_pos / union, true_pos / vol_pred, true_pos / vol_gt
