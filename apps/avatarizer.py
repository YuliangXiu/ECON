import numpy as np
import trimesh
import torch
import os.path as osp
import lib.smplx as smplx
from lib.dataset.mesh_util import SMPLX

smplx_container = SMPLX()

smpl_npy = "./results/github/econ/obj/304e9c4798a8c3967de7c74c24ef2e38_smpl_00.npy"
smplx_param = np.load(smpl_npy, allow_pickle=True).item()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].cpu().view(1, -1)
    # print(key, smplx_param[key].device, smplx_param[key].shape)

smpl_model = smplx.create(
    smplx_container.model_dir,
    model_type="smplx",
    gender="neutral",
    age="adult",
    use_face_contour=False,
    use_pca=False,
    num_betas=200,
    num_expression_coeffs=50,
    ext='pkl')

smpl_out = smpl_model(
    body_pose=smplx_param["body_pose"],
    global_orient=smplx_param["global_orient"],
    # transl=smplx_param["transl"],
    betas=smplx_param["betas"],
    expression=smplx_param["expression"],
    jaw_pose=smplx_param["jaw_pose"],
    left_hand_pose=smplx_param["left_hand_pose"],
    right_hand_pose=smplx_param["right_hand_pose"],
    return_verts=True,
    return_joint_transformation=True,
    return_vertex_transformation=True)

smpl_verts = smpl_out.vertices.detach()[0]
inv_mat = torch.inverse(smpl_out.vertex_transformation.detach()[0])
homo_coord = torch.ones_like(smpl_verts)[..., :1]
smpl_verts = inv_mat @ torch.cat([smpl_verts, homo_coord], dim=1).unsqueeze(-1)
smpl_verts = smpl_verts[:, :3, 0].cpu()

trimesh.Trimesh(smpl_verts, smpl_model.faces).show()
