from lib.common.BNI_utils import (
    depth2arr,
    tensor2arr,
    verts_inverse_transform,
    bilateral_normal_integration,
)

import torch
import os
import trimesh
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import IO
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance


class BNI:

    def __init__(self, dir_path, name, in_tensor, device, mvc=True):

        self.scale = 256.0

        self.normal_front = tensor2arr(in_tensor["normal_F"])
        self.normal_back = tensor2arr(in_tensor["normal_B"])
        self.mask = tensor2arr(in_tensor["image"], True).astype(bool)[..., 0]
        self.depth_front = (depth2arr(in_tensor["depth_F"]) -
                            100.0) * self.scale
        self.depth_back = (100.0 -
                           depth2arr(in_tensor["depth_B"])) * self.scale
        self.depth_mask = depth2arr(in_tensor["depth_F"]) > -1.0
        
        

        # hparam
        self.k = 2
        self.lambda1 = 1e-4
        self._DEFAULT_MIN_TRIANGLE_AREA = 5e-3
        self.name = name

        self.F_B_surface = None
        self.F_B_trimesh = None
        self.device = device
        self.export_dir = dir_path
        
    # @staticmethod
    # def mvp_expand()

    @staticmethod
    def load_all(export_dir, name):

        F_verts = torch.load(os.path.join(export_dir, f"{name}_F_verts.pt"))
        F_faces = torch.load(os.path.join(export_dir, f"{name}_F_faces.pt"))
        B_verts = torch.load(os.path.join(export_dir, f"{name}_B_verts.pt"))
        B_faces = torch.load(os.path.join(export_dir, f"{name}_B_faces.pt"))

        return F_verts, F_faces, B_verts, B_faces

    @staticmethod
    def export_all(F_verts, F_faces, B_verts, B_faces, export_dir, name):

        torch.save(F_verts, os.path.join(export_dir, f"{name}_F_verts.pt"))
        torch.save(F_faces, os.path.join(export_dir, f"{name}_F_faces.pt"))
        torch.save(B_verts, os.path.join(export_dir, f"{name}_B_verts.pt"))
        torch.save(B_faces, os.path.join(export_dir, f"{name}_B_faces.pt"))

    # code: https://github.com/hoshino042/bilateral_normal_integration
    # paper: Bilateral Normal Integration

    def extract_surface(self):

        if not os.path.exists(
                os.path.join(self.export_dir, f"{self.name}_F_verts.pt")):

            F_verts, F_faces = bilateral_normal_integration(
                normal_map=self.normal_front,
                normal_mask=self.mask,
                k=self.k,
                lambda1=self.lambda1,
                depth_map=self.depth_front,
                depth_mask=self.depth_mask,
                label="Front",
            )

            B_verts, B_faces = bilateral_normal_integration(
                normal_map=self.normal_back,
                normal_mask=self.mask,
                k=self.k,
                lambda1=self.lambda1,
                depth_map=self.depth_back,
                depth_mask=self.depth_mask,
                label="Back",
            )
            self.export_all(F_verts, F_faces, B_verts, B_faces,
                            self.export_dir, self.name)
        else:
            F_verts, F_faces, B_verts, B_faces = self.load_all(
                self.export_dir, self.name)

        F_verts = verts_inverse_transform(F_verts, self.scale)
        B_verts = verts_inverse_transform(B_verts, self.scale)

        F_B_verts = torch.cat((F_verts, B_verts), dim=0)
        F_B_faces = torch.cat((F_faces, B_faces + F_faces.max() + 1), dim=0)

        self.F_B_surfaces = Meshes(F_B_verts.float().unsqueeze(0),
                                   F_B_faces.long().unsqueeze(0)).to(
                                       self.device)

        self.F_B_trimesh = trimesh.Trimesh(F_B_verts.float(),
                                           F_B_faces.long(),
                                           process=False,
                                           maintain_order=True)

        IO().save_mesh(
            self.F_B_surfaces,
            os.path.join(self.export_dir, f"{self.name}_F_B_surface.obj"),
        )

    @staticmethod
    def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):
        """
        Computes the distance between a pointcloud and a mesh within a batch.
        Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
        sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

        `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
            to the closest triangular face in mesh and averages across all points in pcl
        `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
            mesh to the closest point in pcl and averages across all faces in mesh.

        The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
        and then averaged across the batch.

        Args:
            meshes: A Meshes data structure containing N meshes
            pcls: A Pointclouds data structure containing N pointclouds
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.

        Returns:
            loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
                between all `(mesh, pcl)` in a batch averaged across the batch.
        """

        if len(meshes) != len(pcls):
            raise ValueError(
                "meshes and pointclouds must be equal sized batches")
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
        point_to_face = _PointFaceDistance.apply(points, points_first_idx,
                                                 tris, tris_first_idx,
                                                 max_points)

        return point_to_face

    def p2s_loss(self, meshes):

        perm = torch.randperm(
            self.F_B_surfaces.verts_packed().shape[0])[:10000]
        random_samples = self.F_B_surfaces.verts_packed()[perm].unsqueeze(0)
        random_samples_dis = self.point_mesh_face_distance(
            meshes, Pointclouds(random_samples))
        # loss = random_samples_dis.mean()
        top_k_index = torch.argsort(random_samples_dis, dim=0,
                                    descending=True)[:5000]
        loss = random_samples_dis[top_k_index].mean()

        return loss
