import torch
from pytorch3d.structures import Meshes, Pointclouds

from lib.common.render_utils import face_vertices
from lib.dataset.Evaluator import point_mesh_distance
from lib.dataset.mesh_util import SMPLX, barycentric_coordinates_of_projection


class PointFeat:
    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        if verts.shape[1] == 10475:
            faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
            mouth_faces = (
                torch.as_tensor(SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(self.Bsize, 1,
                                                                             1).to(self.device)
            )
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts.float()
        self.triangles = face_vertices(self.verts, self.faces)
        self.mesh = Meshes(self.verts, self.faces).to(self.device)

    def query(self, points):

        points = points.float()
        residues, pts_ind = point_mesh_distance(self.mesh, Pointclouds(points), weighted=False)

        closest_triangles = torch.gather(
            self.triangles, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)
        ).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

        feat_normals = face_vertices(self.mesh.verts_normals_padded(), self.faces)
        closest_normals = torch.gather(
            feat_normals, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)
        ).view(-1, 3, 3)
        shoot_verts = ((closest_triangles * bary_weights[:, :, None]).sum(1).unsqueeze(0))

        pts2shoot_normals = points - shoot_verts
        pts2shoot_normals = pts2shoot_normals / torch.norm(pts2shoot_normals, dim=-1, keepdim=True)

        shoot_normals = ((closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0))
        shoot_normals = shoot_normals / torch.norm(shoot_normals, dim=-1, keepdim=True)
        angles = (pts2shoot_normals * shoot_normals).sum(dim=-1).abs()

        return (torch.sqrt(residues).unsqueeze(0), angles)
