import argparse
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
from termcolor import colored

import lib.smplx as smplx
from lib.common.local_affine import register
from lib.dataset.mesh_util import (
    SMPLX,
    export_obj,
    keep_largest,
    poisson,
)
from lib.smplx.lbs import general_lbs

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-uv", action="store_true")
parser.add_argument("-dress", action="store_true")
args = parser.parse_args()

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

# loading SMPL-X and econ objs inferred with ECON
prefix = f"./results/econ/obj/{args.name}"
smpl_path = f"{prefix}_smpl_00.npy"
smplx_param = np.load(smpl_path, allow_pickle=True).item()

# export econ obj with pre-computed normals
econ_path = f"{prefix}_0_full_soups.ply"
econ_obj = trimesh.load(econ_path)
assert econ_obj.vertex_normals.shape[1] == 3
os.makedirs(f"{prefix}/", exist_ok=True)

# align econ with SMPL-X
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
    num_betas=smplx_param["betas"].shape[1],
    num_expression_coeffs=smplx_param["expression"].shape[1],
    ext="pkl",
)

smpl_out_lst = []

# obtain the pose params of T-pose, DA-pose, and the original pose
for pose_type in ["a-pose", "t-pose", "da-pose", "pose"]:
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
            pose_type=pose_type,
        )
    )

# -------------------------- align econ and SMPL-X in DA-pose space ------------------------- #
# 1. find the vertex-correspondence between SMPL-X and econ
# 2. ECON + SMPL-X: posed space --> T-pose space --> DA-pose space
# 3. ECON (w/o hands & over-streched faces) + SMPL-X (w/ hands & registered inpainting parts)
# ------------------------------------------------------------------------------------------- #

smpl_verts = smpl_out_lst[3].vertices.detach()[0]
smpl_tree = cKDTree(smpl_verts.cpu().numpy())
dist, idx = smpl_tree.query(econ_obj.vertices, k=3)

if not osp.exists(f"{prefix}/econ_da.obj") or not osp.exists(f"{prefix}/smpl_da.obj"):

    # t-pose for ECON
    econ_verts = torch.tensor(econ_obj.vertices).float()
    rot_mat_t = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
    homo_coord = torch.ones_like(econ_verts)[..., :1]
    econ_cano_verts = torch.inverse(rot_mat_t) @ torch.cat([econ_verts, homo_coord],
                                                           dim=1).unsqueeze(-1)
    econ_cano_verts = econ_cano_verts[:, :3, 0].cpu()
    econ_cano = trimesh.Trimesh(econ_cano_verts, econ_obj.faces)

    # da-pose for ECON
    rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
    econ_da_verts = rot_mat_da @ torch.cat([econ_cano_verts, homo_coord], dim=1).unsqueeze(-1)
    econ_da = trimesh.Trimesh(econ_da_verts[:, :3, 0].cpu(), econ_obj.faces)

    # da-pose for SMPL-X
    smpl_da = trimesh.Trimesh(
        smpl_out_lst[2].vertices.detach()[0],
        smpl_model.faces,
        maintain_orders=True,
        process=False,
    )
    smpl_da.export(f"{prefix}/smpl_da.obj")

    # ignore parts: hands, front_flame, eyeball
    ignore_vid = np.concatenate([
        smplx_container.smplx_mano_vid,
        smplx_container.smplx_front_flame_vid,
        smplx_container.smplx_eyeball_vid,
    ])

    # a trick to avoid torn dress/skirt
    if args.dress:
        ignore_vid = np.concatenate([ignore_vid, smplx_container.smplx_leg_vid])

    # remove ignore parts from ECON
    econ_da_body = econ_da.copy()
    mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
    econ_da_body.update_faces(mano_mask[econ_da.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()
    econ_da_body = keep_largest(econ_da_body)

    # remove ignore parts from SMPL-X
    register_mask = ~np.isin(np.arange(smpl_da.vertices.shape[0]), ignore_vid)
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

    streched_vid = np.unique(econ_cano.edges[edge_diff > 6])
    mano_mask[streched_vid] = False
    econ_da_body.update_faces(mano_mask[econ_cano.faces].all(axis=1))
    econ_da_body.remove_unreferenced_vertices()

    # stitch the registered SMPL-X body and floating hands to ECON
    econ_da_tree = cKDTree(econ_da.vertices)
    dist, idx = econ_da_tree.query(smpl_da_body.vertices, k=1)
    smpl_da_body.update_faces((dist > 0.02)[smpl_da_body.faces].all(axis=1))
    smpl_da_body.remove_unreferenced_vertices()

    smpl_hand = smpl_da.copy()
    smpl_hand.update_faces(
        smplx_container.smplx_mano_vertex_mask.numpy()[smpl_hand.faces].all(axis=1)
    )
    smpl_hand.remove_unreferenced_vertices()

    # combination of ECON body, SMPL-X side parts, SMPL-X hands
    econ_da = sum([smpl_hand, smpl_da_body, econ_da_body])
    econ_da = poisson(
        econ_da, f"{prefix}/econ_da.obj", depth=10, face_count=1e5, laplacian_remeshing=True
    )
else:
    econ_da = trimesh.load(f"{prefix}/econ_da.obj")
    smpl_da = trimesh.load(f"{prefix}/smpl_da.obj", maintain_orders=True, process=False)

# ---------------------- SMPL-X compatible ECON ---------------------- #
# 1. Find the new vertex-correspondence between NEW ECON and SMPL-X
# 2. Build the new J_regressor, lbs_weights, posedirs
# 3. canonicalize the NEW ECON
# ------------------------------------------------------------------- #

print("Start building the SMPL-X compatible ECON model...")

smpl_tree = cKDTree(smpl_da.vertices)
dist, idx = smpl_tree.query(econ_da.vertices, k=3)
knn_weights = np.exp(-(dist**2))
knn_weights /= knn_weights.sum(axis=1, keepdims=True)

econ_J_regressor = (smpl_model.J_regressor[:, idx] * knn_weights[None]).sum(dim=-1)
econ_lbs_weights = (smpl_model.lbs_weights.T[:, idx] * knn_weights[None]).sum(dim=-1).T

num_posedirs = smpl_model.posedirs.shape[0]
econ_posedirs = ((
    smpl_model.posedirs.view(num_posedirs, -1, 3)[:, idx, :] * knn_weights[None, ..., None]
).sum(dim=-2).view(num_posedirs, -1).float())

econ_J_regressor /= econ_J_regressor.sum(dim=1, keepdims=True).clip(min=1e-10)
econ_lbs_weights /= econ_lbs_weights.sum(dim=1, keepdims=True)

rot_mat_da = smpl_out_lst[2].vertex_transformation.detach()[0][idx[:, 0]]
econ_da_verts = torch.tensor(econ_da.vertices).float()
econ_cano_verts = torch.inverse(rot_mat_da) @ torch.cat([
    econ_da_verts, torch.ones_like(econ_da_verts)[..., :1]
],
                                                        dim=1).unsqueeze(-1)
econ_cano_verts = econ_cano_verts[:, :3, 0].double()

# ----------------------------------------------------
# use original pose to animate ECON reconstruction
# ----------------------------------------------------

rot_mat_pose = smpl_out_lst[3].vertex_transformation.detach()[0][idx[:, 0]]
posed_econ_verts = rot_mat_pose @ torch.cat([
    econ_cano_verts.float(),
    torch.ones_like(econ_cano_verts.float())[..., :1]
],
                                            dim=1).unsqueeze(-1)
posed_econ_verts = posed_econ_verts[:, :3, 0].double()

aligned_econ_verts = posed_econ_verts.detach().cpu().numpy()
aligned_econ_verts += smplx_param["transl"].cpu().numpy()
aligned_econ_verts *= smplx_param["scale"].cpu().numpy() * np.array([1.0, -1.0, -1.0])
econ_pose = trimesh.Trimesh(aligned_econ_verts, econ_da.faces)
assert econ_pose.vertex_normals.shape[1] == 3
econ_pose.export(f"{prefix}/econ_pose.ply")

cache_path = f"{prefix.replace('obj','cache')}"
os.makedirs(cache_path, exist_ok=True)

# -----------------------------------------------------------------
# create UV texture (.obj .mtl .png) from posed ECON reconstruction
# -----------------------------------------------------------------

print("Start Color mapping...")
from PIL import Image
from torchvision import transforms

from lib.common.render import query_color, query_normal_color
from lib.common.render_utils import Pytorch3dRasterizer

# choice 1: pixels to visible regions, normals to invisible regions

if not osp.exists(f"{prefix}/econ_icp_rgb.ply"):
    masked_image = f"./results/econ/png/{args.name}_cloth.png"
    tensor_image = transforms.ToTensor()(Image.open(masked_image))[:, :, :512]

    final_rgb = query_color(
        torch.tensor(econ_pose.vertices).float(),
        torch.tensor(econ_pose.faces).long(),
        ((tensor_image - 0.5) * 2.0).unsqueeze(0).to(device),
        device=device,
        paint_normal=False,
    ).numpy()
    final_rgb[final_rgb == tensor_image[:, 0, 0] * 255.0] = 0.5 * 255.0

    econ_pose.visual.vertex_colors = final_rgb
    econ_pose.export(f"{prefix}/econ_icp_rgb.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_rgb.ply")
    final_rgb = mesh.visual.vertex_colors[:, :3]

# choice 2: normals to all the regions

if not osp.exists(f"{prefix}/econ_icp_normal.ply"):

    file_normal = query_normal_color(
        torch.tensor(econ_pose.vertices).float(),
        torch.tensor(econ_pose.faces).long(),
        device=device,
    ).numpy()

    econ_pose.visual.vertex_colors = file_normal
    econ_pose.export(f"{prefix}/econ_icp_normal.ply")
else:
    mesh = trimesh.load(f"{prefix}/econ_icp_normal.ply")
    file_normal = mesh.visual.vertex_colors[:, :3]

# econ data used for animation and rendering

econ_dict = {
    "v_template": econ_cano_verts.unsqueeze(0),
    "posedirs": econ_posedirs,
    "J_regressor": econ_J_regressor,
    "parents": smpl_model.parents,
    "lbs_weights": econ_lbs_weights,
    "final_rgb": final_rgb,
    "final_normal": file_normal,
    "faces": econ_pose.faces,
}

torch.save(econ_dict, f"{cache_path}/econ.pt")

print(
    colored(
        "If the dress/skirt is torn in `<file_name>/econ_da.obj`, please delete ./file_name and regenerate them with `-dress` \n \
    python -m apps.avatarizer -n <file_name> -dress", "yellow"
    )
)

if args.uv:

    print("Start UV texture generation...")

    # Generate UV coords
    v_np = econ_pose.vertices
    f_np = econ_pose.faces

    vt_cache = osp.join(cache_path, "vt.pt")
    ft_cache = osp.join(cache_path, "ft.pt")

    if osp.exists(vt_cache) and osp.exists(ft_cache):
        vt = torch.load(vt_cache).to(device)
        ft = torch.load(ft_cache).to(device)
    else:
        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        chart_options.max_iterations = 4
        pack_options.resolution = 8192
        pack_options.bruteForce = True
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
        torch.save(vt.cpu(), vt_cache)
        torch.save(ft.cpu(), ft_cache)

    # UV texture rendering
    uv_rasterizer = Pytorch3dRasterizer(image_size=8192, device=device)
    texture_npy = uv_rasterizer.get_texture(
        torch.cat([(vt - 0.5) * 2.0, torch.ones_like(vt[:, :1])], dim=1),
        ft,
        torch.tensor(v_np).unsqueeze(0).float(),
        torch.tensor(f_np).unsqueeze(0).long(),
        torch.tensor(final_rgb).unsqueeze(0).float() / 255.0,
    )

    gray_texture = texture_npy.copy()
    gray_texture[texture_npy.sum(axis=2) == 0.0] = 0.5
    Image.fromarray((gray_texture * 255.0).astype(np.uint8)).save(f"{cache_path}/texture.png")

    # UV mask for TEXTure (https://readpaper.com/paper/4720151447010820097)
    white_texture = texture_npy.copy()
    white_texture[texture_npy.sum(axis=2) == 0.0] = 1.0
    Image.fromarray((white_texture * 255.0).astype(np.uint8)).save(f"{cache_path}/mask.png")

    # generate a-pose vertices
    new_pose = smpl_out_lst[0].full_pose
    new_pose[:, :3] = 0.0

    posed_econ_verts, _ = general_lbs(
        pose=new_pose,
        v_template=econ_cano_verts.unsqueeze(0),
        posedirs=econ_posedirs,
        J_regressor=econ_J_regressor,
        parents=smpl_model.parents,
        lbs_weights=econ_lbs_weights,
    )

    # export mtl file
    with open(f"{cache_path}/material.mtl", "w") as fp:
        fp.write(f"newmtl mat0 \n")
        fp.write(f"Ka 1.000000 1.000000 1.000000 \n")
        fp.write(f"Kd 1.000000 1.000000 1.000000 \n")
        fp.write(f"Ks 0.000000 0.000000 0.000000 \n")
        fp.write(f"Tr 1.000000 \n")
        fp.write(f"illum 1 \n")
        fp.write(f"Ns 0.000000 \n")
        fp.write(f"map_Kd texture.png \n")

    export_obj(posed_econ_verts[0].detach().cpu().numpy(), f_np, vt, ft, f"{cache_path}/mesh.obj")
