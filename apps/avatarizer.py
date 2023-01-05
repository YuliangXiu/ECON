import numpy as np
import trimesh
import torch
import argparse
import os.path as osp
import lib.smplx as smplx
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

from lib.smplx.lbs import general_lbs
from lib.dataset.mesh_util import keep_largest, poisson
from scipy.spatial import cKDTree
from lib.dataset.mesh_util import SMPLX
from lib.common.local_affine import register

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

prefix = f"./results/econ/obj/{args.name}"
smpl_path = f"{prefix}_smpl_00.npy"
econ_path = f"{prefix}_0_full.obj"

smplx_param = np.load(smpl_path, allow_pickle=True).item()
econ_obj = trimesh.load(econ_path)
econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
econ_obj.vertices /= smplx_param["scale"].cpu().numpy()
econ_obj.vertices -= smplx_param["transl"].cpu().numpy()

for key in smplx_param.keys():
    smplx_param[key] = smplx_param[key].cpu().view(1, -1)

smpl_model = smplx.create(
    smplx_container.model_dir,
    model_type="smplx",
    gender="neutral",
    age="adult",
    use_face_contour=False,
    use_pca=False,
    num_betas=200,
    num_expression_coeffs=50,
    ext='pkl'
)

smpl_out_lst = []

for pose_type in ["t-pose", "da-pose", "pose"]:
    smpl_out_lst.append(
        smpl_model(
            body_pose=smplx_param["body_pose"],
            global_orient=smplx_param["global_orient"],
            betas=smplx_param["betas"],
            expression=smplx_param["expression"],
            jaw_pose=smplx_param["jaw_pose"],
            left_hand_pose=smplx_param["left_hand_pose"],
            right_hand_pose=smplx_param["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
            return_joint_transformation=True,
            return_vertex_transformation=True,
            pose_type=pose_type
        )
    )

smpl_verts = smpl_out_lst[2].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=5)

if not osp.exists(f"{prefix}_econ_da.obj") or not osp.exists(f"{prefix}_smpl_da.obj"):

    # t-pose for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    rot_mat_t = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord],
                                                           dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    econ_cano = trimesh.Trimesh(econ_cano_verts, econ_obj.faces)

    # da-pose for ECON
    rot_mat_da = smpl_out_lst[1].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), econ_obj.faces)

    # da-pose for SMPL-X
    smpl_da = trimesh.Trimesh(
        smpl_out_lst[1].vertices.detach()[0], smpl_model.faces, maintain_orders=True, process=False
    )
    smpl_da.export(f"{prefix}_smpl_da.obj")

    # remove hands from ECON for next registeration
    econ_da_body = econ_da.copy()
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    econ_da_body.update_faces(mano_mask[econ_da.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()
    econ_da_body = keep_largest(econ_da_body)

    # remove SMPL-X hand and face
    register_mask = ~np.isin(
        np.arange(smpl_da.vertices.shape[0]),
        np.concatenate([smplx_container.smplx_mano_vid, smplx_container.smplx_front_flame_vid])
    )
    register_mask *= ~smplx_container.eyeball_vertex_mask.bool().numpy()
    smpl_da_body = smpl_da.copy()
    smpl_da_body.update_faces(register_mask[smpl_da.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()
    smpl_da_body = keep_largest(smpl_da_body)

    # upsample the smpl_da_body and do registeration
    smpl_da_body = Meshes(
        verts=[torch.tensor(smpl_da_body.vertices).float()],
        faces=[torch.tensor(smpl_da_body.faces).long()],
    ).to(device)
    sm = SubdivideMeshes(smpl_da_body)
    smpl_da_body = register(econ_da_body, sm(smpl_da_body), device)

    # remove over-streched+hand faces from ECON
    econ_da_body = econ_da.copy()
    edge_before = np.sqrt(
        ((econ_obj.vertices[econ_cano.edges[:, 0]] -
          econ_obj.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1)
    )
    edge_after = np.sqrt(
        ((econ_da.vertices[econ_cano.edges[:, 0]] -
          econ_da.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1)
    )
    edge_diff = edge_after / edge_before.clip(1e-2)
    streched_mask = np.unique(econ_cano.edges[edge_diff > 6])
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    mano_mask[streched_mask] = False
    econ_da_body.update_faces(mano_mask[econ_cano.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()

    # stitch the registered SMPL-X body and floating hands to ECON
    econ_da_tree = cKDTree(econ_da.vertices)
    dist, idx = econ_da_tree.query(smpl_da_body.vertices, k=1)
    smpl_da_body.update_faces((dist > 0.02)[smpl_da_body.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()

    smpl_hand = smpl_da.copy()
    smpl_hand.update_faces(smplx_container.mano_vertex_mask.numpy()[smpl_hand.faces].all(axis=1))
    smpl_hand.remove_unreferenced_vertices()
    econ_da = sum([smpl_hand, smpl_da_body, econ_da_body])
    econ_da = poisson(econ_da, f"{prefix}_econ_da.obj", depth=10, decimation=False)
else:
    econ_da = trimesh.load(f"{prefix}_econ_da.obj")
    smpl_da = trimesh.load(f"{prefix}_smpl_da.obj", maintain_orders=True, process=False)

smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(econ_da.vertices, k=5)
knn_weights = np.exp(-dist**2)
knn_weights /= knn_weights.sum(axis=1, keepdims=True)

econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(dim=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(dim=-1).T

num_posedirs = smpl_model.posedirs.shape[0]
econ_posedirs = (
    smpl_model.posedirs.view(num_posedirs, -1, 3)[:, idx, :] * knn_weights[None, ..., None]
).sum(dim=-2).view(num_posedirs, -1).float()

econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True)

# re-compute da-pose rot_mat for ECON
rot_mat_da = smpl_out_lst[1].vertex_transformation.detach()[0][idx[:, 0]]
econ_da_verts = torch.tensor(econ_da.vertices).float()
econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat(
    [econ_da_verts, torch.ones_like(econ_da_verts)[..., :1]], dim=1
).unsqueeze(-1)
econ_cano_verts = econ_cano_verts[:, :3, 0].double()

# ----------------------------------------------------
# use any SMPL-X pose to animate ECON reconstruction
# ----------------------------------------------------

new_pose = smpl_out_lst[2].full_pose
new_pose[:, :3] = 0.

posed_econ_verts, _ = general_lbs(
    pose=new_pose,
    v_template=econ_cano_verts.unsqueeze(0),
    posedirs=econ_posedirs,
    J_regressor=econ_J_regressor,
    parents=smpl_model.parents,
    lbs_weights=econ_lbs_weights
)

econ_pose = trimesh.Trimesh(posed_econ_verts[0].detach(), econ_da.faces)
econ_pose.export(f"{prefix}_econ_pose.obj")
