from lib.common.BNI_utils import (verts_inverse_transform, depth_inverse_transform,
                                  double_side_bilateral_normal_integration, mean_value_cordinates,
                                  find_contour, depth2png, dispCorres, repeat_pts, get_dst_mat)

import torch
import os, cv2
import trimesh
import numpy as np
import os.path as osp
from scipy.optimize import linear_sum_assignment
from pytorch3d.structures import Meshes
from pytorch3d.io import IO


class BNI:

    def __init__(self, dir_path, name, BNI_dict, cfg, device):

        self.scale = 256.0
        self.cfg = cfg
        self.name = name

        self.normal_front = BNI_dict["normal_F"]
        self.normal_back = BNI_dict["normal_B"]
        self.mask = BNI_dict["mask"]

        self.depth_front = BNI_dict["depth_F"]
        self.depth_back = BNI_dict["depth_B"]
        self.depth_mask = BNI_dict["depth_mask"]

        # hparam:
        # k --> smaller, keep continuity
        # lambda --> larger, more depth-awareness

        # self.k = 1e-3
        # self.lambda1 = 1e-2

        self.k = self.cfg.k
        self.lambda1 = self.cfg.lambda1
        self.boundary_consist = self.cfg.boundary_consist

        self.F_B_surface = None
        self.F_B_trimesh = None
        self.F_depth = None
        self.B_depth = None

        self.device = device
        self.export_dir = dir_path

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface(self, verbose=True):

        bni_result = double_side_bilateral_normal_integration(
            normal_front=self.normal_front,
            normal_back=self.normal_back,
            normal_mask=self.mask,
            depth_front=self.depth_front * self.scale,
            depth_back=self.depth_back * self.scale,
            depth_mask=self.depth_mask,
            k=self.k,
            lambda_normal_back=1.0,
            lambda_depth_front=self.lambda1,
            lambda_depth_back=self.lambda1,
            lambda_boundary_consistency=self.boundary_consist,
        )

        F_verts = verts_inverse_transform(bni_result["F_verts"], self.scale)
        B_verts = verts_inverse_transform(bni_result["B_verts"], self.scale)

        self.F_depth = depth_inverse_transform(bni_result["F_depth"], self.scale)
        self.B_depth = depth_inverse_transform(bni_result["B_depth"], self.scale)

        F_B_verts = torch.cat((F_verts, B_verts), dim=0)
        F_B_faces = torch.cat(
            (bni_result["F_faces"], bni_result["B_faces"] + bni_result["F_faces"].max() + 1), dim=0)

        self.F_B_trimesh = trimesh.Trimesh(F_B_verts.float(),
                                           F_B_faces.long(),
                                           process=False,
                                           maintain_order=True)

        self.F_trimesh = trimesh.Trimesh(F_verts.float(),
                                         bni_result["F_faces"].long(),
                                         process=False,
                                         maintain_order=True)

        self.B_trimesh = trimesh.Trimesh(B_verts.float(),
                                         bni_result["B_faces"].long(),
                                         process=False,
                                         maintain_order=True)
