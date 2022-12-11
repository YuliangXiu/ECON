import numpy as np
import trimesh
import torch
import os.path as osp
import lib.smplx as smplx
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

from lib.smplx.lbs import general_lbs
from lib.dataset.mesh_util import keep_largest, poisson
from scipy.spatial import cKDTree
from lib.dataset.mesh_util import SMPLX
from lib.common.local_affine import register

smplx_container = SMPLX()
device = torch.device("cuda:0")

prefix = "./results/github/econ/obj/304e9c4798a8c3967de7c74c24ef2e38"
smpl_path = f"{prefix}_smpl_00.npy"
econ_path = f"{prefix}_0_full.obj"

smplx_param = np.load(smpl_path, allow_pickle=True).item()
econ_obj = trimesh.load(econ_path)
econ_obj.vertices *= np.array([1.0, -1.0, -1.0])
econ_obj.vertices /= smplx_param["scale"].cpu().numpy()
econ_obj.vertices -= smplx_param["transl"].cpu().numpy()

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
    betas=smplx_param["betas"],
    expression=smplx_param["expression"],
    jaw_pose=smplx_param["jaw_pose"],
    left_hand_pose=smplx_param["left_hand_pose"],
    right_hand_pose=smplx_param["right_hand_pose"],
    return_verts=True,
    return_full_pose=True,
    return_joint_transformation=True,
    return_vertex_transformation=True)

smpl_verts = smpl_out.vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=5)

if not osp.exists(f"{prefix}_econ_cano.obj") or not osp.exists(f"{prefix}_smpl_cano.obj"):

    # canonicalize for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    inv_mat = torch.inverse(smpl_out.vertex_transformation.detach()[0][idx[:, 0]])
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = inv_mat @ torch.cat([econ_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    econ_cano = trimesh.Trimesh(econ_cano_verts, econ_obj.faces)

    # canonicalize for SMPL-X
    inv_mat = torch.inverse(smpl_out.vertex_transformation.detach()[0])
    homo_coord = torch.ones_like(smpl_verts)[..., :1]
    smpl_cano_verts = inv_mat @ torch.cat([smpl_verts, homo_coord], dim=1).unsqueeze(-1)
    smpl_cano_verts = smpl_cano_verts[:, :3, 0].cpu()
    smpl_cano = trimesh.Trimesh(smpl_cano_verts, smpl_model.faces, maintain_orders=True, process=False)
    smpl_cano.export(f"{prefix}_smpl_cano.obj")

    # remove hands from ECON for next registeration
    econ_cano_body = econ_cano.copy()
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    econ_cano_body.update_faces(mano_mask[econ_cano.faces].all(axis=1))
    econ_cano_body.remove_unreferenced_vertices()
    econ_cano_body = keep_largest(econ_cano_body)

    # remove SMPL-X hand and face
    register_mask = ~np.isin(
        np.arange(smpl_cano_verts.shape[0]),
        np.concatenate([smplx_container.smplx_mano_vid, smplx_container.smplx_front_flame_vid]))
    register_mask *= ~smplx_container.eyeball_vertex_mask.bool().numpy()
    smpl_cano_body = smpl_cano.copy()
    smpl_cano_body.update_faces(register_mask[smpl_cano.faces].all(axis=1))
    smpl_cano_body.remove_unreferenced_vertices()
    smpl_cano_body = keep_largest(smpl_cano_body)

    # upsample the smpl_cano_body and do registeration
    smpl_cano_body = Meshes(
        verts=[torch.tensor(smpl_cano_body.vertices).float()],
        faces=[torch.tensor(smpl_cano_body.faces).long()],
    ).to(device)
    sm = SubdivideMeshes(smpl_cano_body)
    smpl_cano_body = register(econ_cano_body, sm(smpl_cano_body), device)

    # remove over-streched+hand faces from ECON
    econ_cano_body = econ_cano.copy()
    edge_before = np.sqrt(
        ((econ_obj.vertices[econ_cano.edges[:, 0]] - econ_obj.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1))
    edge_after = np.sqrt(
        ((econ_cano.vertices[econ_cano.edges[:, 0]] - econ_cano.vertices[econ_cano.edges[:, 1]])**2).sum(axis=1))
    edge_diff = edge_after / edge_before.clip(1e-2)
    streched_mask = np.unique(econ_cano.edges[edge_diff > 6])
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    mano_mask[streched_mask] = False
    econ_cano_body.update_faces(mano_mask[econ_cano.faces].all(axis=1))
    econ_cano_body.remove_unreferenced_vertices()

    # stitch the registered SMPL-X body and floating hands to ECON
    econ_cano_tree = cKDTree(econ_cano.vertices)
    dist, idx = econ_cano_tree.query(smpl_cano_body.vertices, k=1)
    smpl_cano_body.update_faces((dist > 0.02)[smpl_cano_body.faces].all(axis=1))
    smpl_cano_body.remove_unreferenced_vertices()

    smpl_hand = smpl_cano.copy()
    smpl_hand.update_faces(smplx_container.mano_vertex_mask.numpy()[smpl_hand.faces].all(axis=1))
    smpl_hand.remove_unreferenced_vertices()
    econ_cano = sum([smpl_hand, smpl_cano_body, econ_cano_body])
    econ_cano = poisson(econ_cano, f"{prefix}_econ_cano.obj")
else:
    econ_cano = trimesh.load(f"{prefix}_econ_cano.obj")
    smpl_cano = trimesh.load(f"{prefix}_smpl_cano.obj", maintain_orders=True, process=False)

smpl_tree = cKDTree(smpl_cano.vertices)
dist, idx = smpl_tree.query(econ_cano.vertices, k=2)
knn_weights = np.exp(-dist**2)
knn_weights /= knn_weights.sum(axis=1, keepdims=True)
econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(axis=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(axis=-1).T
econ_J_regressor /= econ_J_regressor.sum(axis=1, keepdims=True)
econ_lbs_weights /= econ_lbs_weights.sum(axis=1, keepdims=True)

posed_econ_verts, _ = general_lbs(
    pose=smpl_out.full_pose,
    v_template=torch.tensor(econ_cano.vertices).unsqueeze(0),
    J_regressor=econ_J_regressor,
    parents=smpl_model.parents,
    lbs_weights=econ_lbs_weights)

econ_pose = trimesh.Trimesh(posed_econ_verts[0].detach(), econ_cano.faces)
econ_pose.export(f"{prefix}_econ_pose.obj")