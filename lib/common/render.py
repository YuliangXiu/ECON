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

import math
import os

import cv2
import numpy as np
import torch
from PIL import ImageColor
from pytorch3d.renderer import (
    AlphaCompositor,
    BlendParams,
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from termcolor import colored
from tqdm import tqdm

import lib.common.render_utils as util
from lib.common.imutils import blend_rgb_norm
from lib.dataset.mesh_util import get_visibility


def image2vid(images, vid_path):

    os.makedirs(os.path.dirname(vid_path), exist_ok=True)

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(vid_path, fourcc, len(images) / 5.0, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


def query_color(verts, faces, image, device, paint_normal=True):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]]).flatten()
    uv = xy.unsqueeze(0).unsqueeze(2)    # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = ((
        torch.nn.functional.grid_sample(image, uv, align_corners=True)[0, :, :, 0].permute(1, 0) +
        1.0
    ) * 0.5 * 255.0)
    if paint_normal:
        colors[visibility == 0.0] = ((
            Meshes(verts.unsqueeze(0), faces.unsqueeze(0)).verts_normals_padded().squeeze(0) + 1.0
        ) * 0.5 * 255.0)[visibility == 0.0]
    else:
        colors[visibility == 0.0] = (torch.tensor([0.5, 0.5, 0.5]) * 255.0).to(device)

    return colors.detach().cpu()


def query_normal_color(verts, faces, device):
    """query normal colors

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    colors = (
        (Meshes(verts.unsqueeze(0), faces.unsqueeze(0)).verts_normals_padded().squeeze(0) + 1.0) *
        0.5 * 255.0
    )

    return colors.detach().cpu()


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images


class Render:
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 100.0
        self.scale = 100.0
        self.mesh_y_center = 0.0

        # speed control
        self.fps = 30
        self.step = 3

        self.cam_pos = {
            "front":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (0, self.mesh_y_center, -self.dis),
            ]), "frontback":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (0, self.mesh_y_center, -self.dis),
            ]), "four":
            torch.tensor([
                (0, self.mesh_y_center, self.dis),
                (self.dis, self.mesh_y_center, 0),
                (0, self.mesh_y_center, -self.dis),
                (-self.dis, self.mesh_y_center, 0),
            ]), "around":
            torch.tensor([(
                100.0 * math.cos(np.pi / 180 * angle), self.mesh_y_center,
                100.0 * math.sin(np.pi / 180 * angle)
            ) for angle in range(0, 360, self.step)])
        }

        self.type = "color"

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None

        self.uv_rasterizer = util.Pytorch3dRasterizer(self.size)

    def get_camera_batch(self, type="four", idx=None):

        if idx is None:
            idx = np.arange(len(self.cam_pos[type]))

        R, T = look_at_view_transform(
            eye=self.cam_pos[type][idx],
            at=((0, self.mesh_y_center, 0), ),
            up=((0, 1, 0), ),
        )

        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3), ) * len(R),
        )

        return cameras

    def init_renderer(self, camera, type="mesh", bg="gray"):

        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

        if ("mesh" in type) or ("depth" in type) or ("rgb" in type):

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                bin_size=-1,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(cameras=camera, raster_settings=self.raster_settings_mesh)

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(blend_params=blendparam),
            )

        elif type == "mask":

            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50,
                bin_size=-1,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette
            )
            self.renderer = MeshRenderer(
                rasterizer=self.silhouetteRas, shader=SoftSilhouetteShader()
            )

        elif type == "pointcloud":
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size, radius=0.006, points_per_pixel=10
            )

            self.pcdRas = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)),
            )

    def load_meshes(self, verts, faces):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3] / [B,N,3]): array or tensor
            faces ([N,3]/ [B,N,3]): array or tensor
        """

        if isinstance(verts, list):
            V_lst = []
            F_lst = []
            for V, F in zip(verts, faces):
                if not torch.is_tensor(V):
                    V_lst.append(torch.tensor(V).float().to(self.device))
                    F_lst.append(torch.tensor(F).long().to(self.device))
                else:
                    V_lst.append(V.float().to(self.device))
                    F_lst.append(F.long().to(self.device))
            self.meshes = Meshes(V_lst, F_lst).to(self.device)
        else:
            # array or tensor
            if not torch.is_tensor(verts):
                verts = torch.tensor(verts)
                faces = torch.tensor(faces)
            if verts.ndimension() == 2:
                verts = verts.float().unsqueeze(0).to(self.device)
                faces = faces.long().unsqueeze(0).to(self.device)
            if verts.shape[0] != faces.shape[0]:
                faces = faces.repeat(len(verts), 1, 1).to(self.device)
            self.meshes = Meshes(verts, faces).to(self.device)

        # texture only support single mesh
        if len(self.meshes) == 1:
            self.meshes.textures = TexturesVertex(
                verts_features=(self.meshes.verts_normals_padded() + 1.0) * 0.5
            )

    def get_image(self, cam_type="frontback", type="rgb", bg="gray"):

        self.init_renderer(self.get_camera_batch(cam_type), type, bg)

        img_lst = []

        for mesh_id in range(len(self.meshes)):

            current_mesh = self.meshes[mesh_id]
            current_mesh.textures = TexturesVertex(
                verts_features=(current_mesh.verts_normals_padded() + 1.0) * 0.5
            )

            if type == "depth":
                fragments = self.meshRas(current_mesh.extend(len(self.cam_pos[cam_type])))
                images = fragments.zbuf[..., 0]

            elif type == "rgb":
                images = self.renderer(current_mesh.extend(len(self.cam_pos[cam_type])))
                images = (images[:, :, :, :3].permute(0, 3, 1, 2) - 0.5) * 2.0

            elif type == "mask":
                images = self.renderer(current_mesh.extend(len(self.cam_pos[cam_type])))[:, :, :, 3]
            else:
                print(f"unknown {type}")

            if cam_type == 'frontback':
                images[1] = torch.flip(images[1], dims=(-1, ))

            # images [N_render, 3, res, res]
            img_lst.append(images.unsqueeze(1))

        # meshes [N_render, N_mesh, 3, res, res]
        meshes = torch.cat(img_lst, dim=1)

        return list(meshes)

    def get_rendered_video_multi(self, data, save_path):

        height, width = data["img_raw"].shape[2:]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            save_path,
            fourcc,
            self.fps,
            (width * 3, int(height)),
        )

        pbar = tqdm(range(len(self.meshes)))
        pbar.set_description(colored(f"Normal Rendering {os.path.basename(save_path)}...", "blue"))

        mesh_renders = []    #[(N_cam, 3, res, res)*N_mesh]

        # render all the normals
        for mesh_id in pbar:

            current_mesh = self.meshes[mesh_id]
            current_mesh.textures = TexturesVertex(
                verts_features=(current_mesh.verts_normals_padded() + 1.0) * 0.5
            )

            norm_lst = []

            for batch_cams_idx in np.array_split(np.arange(len(self.cam_pos["around"])), 12):

                batch_cams = self.get_camera_batch(type='around', idx=batch_cams_idx)

                self.init_renderer(batch_cams, "mesh", "gray")

                norm_lst.append(
                    self.renderer(current_mesh.extend(len(batch_cams_idx))
                                 )[..., :3].permute(0, 3, 1, 2)
                )
            mesh_renders.append(torch.cat(norm_lst).detach().cpu())

        # generate video frame by frame
        pbar = tqdm(range(len(self.cam_pos["around"])))
        pbar.set_description(colored(f"Video Exporting {os.path.basename(save_path)}...", "blue"))

        for cam_id in pbar:
            img_raw = data["img_raw"]
            num_obj = len(mesh_renders) // 2
            img_smpl = blend_rgb_norm((torch.stack(mesh_renders)[:num_obj, cam_id] - 0.5) * 2.0,
                                      data)
            img_cloth = blend_rgb_norm((torch.stack(mesh_renders)[num_obj:, cam_id] - 0.5) * 2.0,
                                       data)
            final_img = torch.cat([img_raw, img_smpl, img_cloth],
                                  dim=-1).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

            video.write(final_img[:, :, ::-1])

        video.release()
