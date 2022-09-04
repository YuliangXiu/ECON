from pytorch3d.structures import Meshes
import torch.nn.functional as F
import torch
from lib.common.render_utils import face_vertices
from lib.dataset.mesh_util import SMPLX, barycentric_coordinates_of_projection
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance


class PointFeat:

    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.mesh = Meshes(verts, faces)
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        if verts.shape[1] == 10475:
            faces = faces[:, ~SMPLX().smplx_eyeball_fid]
            mouth_faces = torch.as_tensor(SMPLX().smplx_mouth_fid).unsqueeze(
                0).repeat(self.Bsize, 1, 1).to(self.device)
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts
        self.triangles = face_vertices(self.verts, self.faces)

    def query(self, points, feats={}):

        # points [B, N, 3]
        # feats {'feat_name': [B, N, C]}

        del_keys = ['smpl_verts', 'smpl_faces', 'smpl_joint']

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        out_dict = {}

        for feat_key in feats.keys():

            if feat_key in del_keys:
                continue
            
            elif feats[feat_key] is not None:
                feat_arr = feats[feat_key]
                feat_dim = feat_arr.shape[-1]
                feat_tri = face_vertices(feat_arr, self.faces)
                closest_feats = torch.gather(
                    feat_tri, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, feat_dim)).view(-1, 3, feat_dim)
                pts_feats = (
                    closest_feats*bary_weights[:, :, None]).sum(1).unsqueeze(0)
                out_dict[feat_key.split("_")[1]] = pts_feats
                
            else:
                out_dict[feat_key.split("_")[1]] = None

        if 'sdf' in out_dict.keys():
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
            pts_signs = 2.0 * \
                (check_sign(self.verts, self.faces[0], points).float() - 0.5)
            pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
            out_dict['sdf'] = pts_sdf

        if 'vis' in out_dict.keys():
            out_dict['vis'] = out_dict['vis'].ge(1e-1).float()

        if 'norm' in out_dict.keys():
            pts_norm = out_dict['norm'] * \
                torch.tensor([-1.0, 1.0, -1.0]).to(self.device)
            out_dict['norm'] = F.normalize(pts_norm, dim=2)
            
        if 'cmap' in out_dict.keys():
            out_dict['cmap'] = out_dict['cmap'].clamp_(min=0.0, max=1.0)

        for out_key in out_dict.keys():
            out_dict[out_key] = out_dict[out_key].view(
                self.Bsize, -1, out_dict[out_key].shape[-1])

        return out_dict
