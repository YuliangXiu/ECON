import imp
import os
from pickle import NONE
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import trimesh
import numpy as np
# import neural_renderer as nr
from skimage.transform import resize
from torchvision.utils import make_grid
import torch.nn.functional as F

from models.smpl import get_smpl_faces, get_model_faces, get_model_tpose
from utils.densepose_methods import DensePoseMethods
from core import constants, path_config
import json
from .geometry import convert_to_full_img_cam
from utils.imutils import crop

try:
    import math
    import pyrender
    from pyrender.constants import RenderFlags
except:
    pass
try:
    from opendr.renderer import ColoredRenderer
    from opendr.lighting import LambertianPointLight, SphericalHarmonics
    from opendr.camera import ProjectPoints
except:
    pass

from pytorch3d.structures.meshes import Meshes
# from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PerspectiveCameras, AmbientLights, PointLights,
    RasterizationSettings, BlendParams, MeshRenderer, MeshRasterizer, SoftPhongShader,
    SoftSilhouetteShader, HardPhongShader, HardGouraudShader, HardFlatShader, TexturesVertex
)

import logging

logger = logging.getLogger(__name__)


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(
        self, scale, translation, znear=pyrender.camera.DEFAULT_Z_NEAR, zfar=None, name=None
    ):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class PyRenderer:
    def __init__(
        self, resolution=(224, 224), orig_img=False, wireframe=False, scale_ratio=1., vis_ratio=1.
    ):
        self.resolution = (resolution[0] * scale_ratio, resolution[1] * scale_ratio)
        # self.scale_ratio = scale_ratio

        self.faces = {
            'smplx': get_model_faces('smplx'),
            'smpl': get_model_faces('smpl'),
        #   'mano': get_model_faces('mano'),
        #   'flame': get_model_faces('flame'),
        }
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0], viewport_height=self.resolution[1], point_size=1.0
        )

        self.vis_ratio = vis_ratio

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=1)

        yrot = np.radians(120)    # angle of lights

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        spot_l = pyrender.SpotLight(
            color=np.ones(3), intensity=15.0, innerConeAngle=np.pi / 3, outerConeAngle=np.pi / 2
        )

        light_pose[:3, 3] = [1, 2, 2]
        self.scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [-1, 2, 2]
        self.scene.add(spot_l, pose=light_pose)

        # light_pose[:3, 3] = [-2, 2, 0]
        # self.scene.add(spot_l, pose=light_pose)

        # light_pose[:3, 3] = [-2, 2, 0]
        # self.scene.add(spot_l, pose=light_pose)

        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
        # 'purple': np.array([0.5, 0.5, 0.7]),
            'purple': np.array([0.55, 0.4, 0.9]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }

    def __call__(
        self,
        verts,
        faces=None,
        img=np.zeros((224, 224, 3)),
        cam=np.array([1, 0, 0]),
        focal_length=[5000, 5000],
        camera_rotation=np.eye(3),
        crop_info=None,
        angle=None,
        axis=None,
        mesh_filename=None,
        color_type=None,
        color=[1.0, 1.0, 0.9],
        iwp_mode=True,
        crop_img=True,
        mesh_type='smpl',
        scale_ratio=1.,
        rgba_mode=False
    ):

        if faces is None:
            faces = self.faces[mesh_type]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        cam = cam.copy()
        if iwp_mode:
            resolution = np.array(img.shape[:2]) * scale_ratio
            if len(cam) == 4:
                sx, sy, tx, ty = cam
                # sy = sx
                camera_translation = np.array(
                    [tx, ty, 2 * focal_length[0] / (resolution[0] * sy + 1e-9)]
                )
            elif len(cam) == 3:
                sx, tx, ty = cam
                sy = sx
                camera_translation = np.array(
                    [-tx, ty, 2 * focal_length[0] / (resolution[0] * sy + 1e-9)]
                )
            render_res = resolution
            self.renderer.viewport_width = render_res[1]
            self.renderer.viewport_height = render_res[0]
        else:
            if crop_info['opt_cam_t'] is None:
                camera_translation = convert_to_full_img_cam(
                    pare_cam=cam[None],
                    bbox_height=crop_info['bbox_scale'] * 200.,
                    bbox_center=crop_info['bbox_center'],
                    img_w=crop_info['img_w'],
                    img_h=crop_info['img_h'],
                    focal_length=focal_length[0],
                )
            else:
                camera_translation = crop_info['opt_cam_t']
            if torch.is_tensor(camera_translation):
                camera_translation = camera_translation[0].cpu().numpy()
            camera_translation = camera_translation.copy()
            camera_translation[0] *= -1
            if 'img_h' in crop_info and 'img_w' in crop_info:
                render_res = (int(crop_info['img_h'][0]), int(crop_info['img_w'][0]))
            else:
                render_res = img.shape[:2] if type(img) is not list else img[0].shape[:2]
            self.renderer.viewport_width = render_res[1]
            self.renderer.viewport_height = render_res[0]
            camera_rotation = camera_rotation.T
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length[0], fy=focal_length[1], cx=render_res[1] / 2., cy=render_res[0] / 2.
        )

        if color_type != None:
            color = self.colors_dict[color_type]

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME | RenderFlags.SHADOWS_SPOT
        else:
            render_flags = RenderFlags.RGBA | RenderFlags.SHADOWS_SPOT

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        if crop_info is not None and crop_img:
            crop_res = img.shape[:2]
            rgb, _, _ = crop(rgb, crop_info['bbox_center'][0], crop_info['bbox_scale'][0], crop_res)

        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]

        image_list = [img] if type(img) is not list else img

        return_img = []
        for item in image_list:
            if scale_ratio != 1:
                orig_size = item.shape[:2]
                item = resize(
                    item, (orig_size[0] * scale_ratio, orig_size[1] * scale_ratio),
                    anti_aliasing=True
                )
                item = (item * 255).astype(np.uint8)
            output_img = rgb[:, :, :-1] * valid_mask * self.vis_ratio + (
                1 - valid_mask * self.vis_ratio
            ) * item
            # output_img[valid_mask < 0.5] = item[valid_mask < 0.5]
            # if scale_ratio != 1:
            #     output_img = resize(output_img, (orig_size[0], orig_size[1]), anti_aliasing=True)
            if rgba_mode:
                output_img_rgba = np.zeros((output_img.shape[0], output_img.shape[1], 4))
                output_img_rgba[:, :, :3] = output_img
                output_img_rgba[:, :, 3][valid_mask[:, :, 0]] = 255
                output_img = output_img_rgba.astype(np.uint8)
            image = output_img.astype(np.uint8)
            return_img.append(image)
            return_img.append(item)

        if type(img) is not list:
            # if scale_ratio == 1:
            return_img = return_img[0]

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return return_img


class OpenDRenderer:
    def __init__(self, resolution=(224, 224), ratio=1):
        self.resolution = (resolution[0] * ratio, resolution[1] * ratio)
        self.ratio = ratio
        self.focal_length = 5000.
        self.K = np.array(
            [
                [self.focal_length, 0., self.resolution[1] / 2.],
                [0., self.focal_length, self.resolution[0] / 2.], [0., 0., 1.]
            ]
        )
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.5, 0.5, 0.7]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
        self.renderer = ColoredRenderer()
        self.faces = get_smpl_faces()

    def reset_res(self, resolution):
        self.resolution = (resolution[0] * self.ratio, resolution[1] * self.ratio)
        self.K = np.array(
            [
                [self.focal_length, 0., self.resolution[1] / 2.],
                [0., self.focal_length, self.resolution[0] / 2.], [0., 0., 1.]
            ]
        )

    def __call__(
        self,
        verts,
        faces=None,
        color=None,
        color_type='white',
        R=None,
        mesh_filename=None,
        img=np.zeros((224, 224, 3)),
        cam=np.array([1, 0, 0]),
        rgba=False,
        addlight=True
    ):
        '''Render mesh using OpenDR
        verts: shape - (V, 3)
        faces: shape - (F, 3)
        img: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        axis: rotate along with X/Y/Z axis (by angle)
        R: rotation matrix (used to manipulate verts) shape - [3, 3]
        Return:
            rendered img: shape - (224, 224, 3), range - [0, 255] (np.uint8)
        '''
        ## Create OpenDR renderer
        rn = self.renderer
        h, w = self.resolution
        K = self.K

        f = np.array([K[0, 0], K[1, 1]])
        c = np.array([K[0, 2], K[1, 2]])

        if faces is None:
            faces = self.faces
        if len(cam) == 4:
            t = np.array([cam[2], cam[3], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
        elif len(cam) == 3:
            t = np.array([cam[1], cam[2], 2 * K[0, 0] / (w * cam[0] + 1e-9)])

        rn.camera = ProjectPoints(rt=np.array([0, 0, 0]), t=t, f=f, c=c, k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 1000., 'width': w, 'height': h}

        albedo = np.ones_like(verts) * .9

        if color is not None:
            color0 = np.array(color)
            color1 = np.array(color)
            color2 = np.array(color)
        elif color_type == 'white':
            color0 = np.array([1., 1., 1.])
            color1 = np.array([1., 1., 1.])
            color2 = np.array([0.7, 0.7, 0.7])
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]
        else:
            color0 = self.colors_dict[color_type] * 1.2
            color1 = self.colors_dict[color_type] * 1.2
            color2 = self.colors_dict[color_type] * 1.2
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]

        # render_smpl = rn.r
        if R is not None:
            assert R.shape == (3, 3), "Shape of rotation matrix should be (3, 3)"
            verts = np.dot(verts, R)

        rn.set(v=verts, f=faces, vc=color, bgcolor=np.zeros(3))

        if addlight:
            yrot = np.radians(120)    # angle of lights
            # # 1. 1. 0.7
            rn.vc = LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-200, -100, -100]), yrot),
                vc=albedo,
                light_color=color0
            )

            # Construct Left Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([800, 10, 300]), yrot),
                vc=albedo,
                light_color=color1
            )

            # Construct Right Light
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
                vc=albedo,
                light_color=color2
            )

        rendered_image = rn.r
        visibility_image = rn.visibility_image

        image_list = [img] if type(img) is not list else img

        return_img = []
        for item in image_list:
            if self.ratio != 1:
                img_resized = resize(
                    item, (item.shape[0] * self.ratio, item.shape[1] * self.ratio),
                    anti_aliasing=True
                )
            else:
                img_resized = item / 255.

            try:
                img_resized[visibility_image != (2**32 - 1)
                           ] = rendered_image[visibility_image != (2**32 - 1)]
            except:
                logger.warning('Can not render mesh.')

            img_resized = (img_resized * 255).astype(np.uint8)
            res = img_resized

            if rgba:
                img_resized_rgba = np.zeros((img_resized.shape[0], img_resized.shape[1], 4))
                img_resized_rgba[:, :, :3] = img_resized
                img_resized_rgba[:, :, 3][visibility_image != (2**32 - 1)] = 255
                res = img_resized_rgba.astype(np.uint8)
            return_img.append(res)

        if type(img) is not list:
            return_img = return_img[0]

        return return_img


#  https://github.com/classner/up/blob/master/up_tools/camera.py
def rotateY(points, angle):
    """Rotate all points in a 2D array around the y axis."""
    ry = np.array(
        [[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0.,
                                                            np.cos(angle)]]
    )
    return np.dot(points, ry)


def rotateX(points, angle):
    """Rotate all points in a 2D array around the x axis."""
    rx = np.array(
        [[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle),
                                                             np.cos(angle)]]
    )
    return np.dot(points, rx)


def rotateZ(points, angle):
    """Rotate all points in a 2D array around the z axis."""
    rz = np.array(
        [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    )
    return np.dot(points, rz)


class IUV_Renderer(object):
    def __init__(
        self,
        focal_length=5000.,
        orig_size=224,
        output_size=56,
        mode='iuv',
        device=torch.device('cuda'),
        mesh_type='smpl'
    ):

        self.focal_length = focal_length
        self.orig_size = orig_size
        self.output_size = output_size

        if mode in ['iuv']:
            if mesh_type == 'smpl':
                DP = DensePoseMethods()

                vert_mapping = DP.All_vertices.astype('int64') - 1
                self.vert_mapping = torch.from_numpy(vert_mapping)

                faces = DP.FacesDensePose
                faces = faces[None, :, :]
                self.faces = torch.from_numpy(
                    faces.astype(np.int32)
                )    # [1, 13774, 3], torch.int32

                num_part = float(np.max(DP.FaceIndices))
                self.num_part = num_part

                dp_vert_pid_fname = 'data/dp_vert_pid.npy'
                if os.path.exists(dp_vert_pid_fname):
                    dp_vert_pid = list(np.load(dp_vert_pid_fname))
                else:
                    print('creating data/dp_vert_pid.npy')
                    dp_vert_pid = []
                    for v in range(len(vert_mapping)):
                        for i, f in enumerate(DP.FacesDensePose):
                            if v in f:
                                dp_vert_pid.append(DP.FaceIndices[i])
                                break
                    np.save(dp_vert_pid_fname, np.array(dp_vert_pid))

                textures_vts = np.array(
                    [
                        (dp_vert_pid[i] / num_part, DP.U_norm[i], DP.V_norm[i])
                        for i in range(len(vert_mapping))
                    ]
                )
                self.textures_vts = torch.from_numpy(
                    textures_vts[None].astype(np.float32)
                )    # (1, 7829, 3)
        elif mode == 'pncc':
            self.vert_mapping = None
            self.faces = torch.from_numpy(
                get_model_faces(mesh_type)[None].astype(np.int32)
            )    #  mano: torch.Size([1, 1538, 3])
            textures_vts = get_model_tpose(mesh_type).unsqueeze(
                0
            )    # mano: torch.Size([1, 778, 3])

            texture_min = torch.min(textures_vts) - 0.001
            texture_range = torch.max(textures_vts) - texture_min + 0.001
            self.textures_vts = (textures_vts - texture_min) / texture_range
        elif mode in ['seg']:
            self.vert_mapping = None
            body_model = 'smpl'

            self.faces = torch.from_numpy(get_smpl_faces().astype(np.int32)[None])

            with open(
                os.path.join(
                    path_config.SMPL_MODEL_DIR, '{}_vert_segmentation.json'.format(body_model)
                ), 'rb'
            ) as json_file:
                smpl_part_id = json.load(json_file)

            v_id = []
            for k in smpl_part_id.keys():
                v_id.extend(smpl_part_id[k])

            v_id = torch.tensor(v_id)
            n_verts = len(torch.unique(v_id))
            num_part = len(constants.SMPL_PART_ID.keys())
            self.num_part = num_part

            seg_vert_pid = np.zeros(n_verts)
            for k in smpl_part_id.keys():
                seg_vert_pid[smpl_part_id[k]] = constants.SMPL_PART_ID[k]

            print('seg_vert_pid', seg_vert_pid.shape)
            textures_vts = seg_vert_pid[:, None].repeat(3, axis=1) / num_part
            print('textures_vts', textures_vts.shape)
            # textures_vts = np.array(
            #     [(seg_vert_pid[i] / num_part,) * 3 for i in
            #     range(n_verts)])
            self.textures_vts = torch.from_numpy(textures_vts[None].astype(np.float32))

        K = np.array(
            [
                [self.focal_length, 0., self.orig_size / 2.],
                [0., self.focal_length, self.orig_size / 2.], [0., 0., 1.]
            ]
        )

        R = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

        t = np.array([0, 0, 5])

        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        camK = F.pad(self.K, (0, 1, 0, 1), "constant", 0)
        camK[:, 2, 2] = 0
        camK[:, 3, 2] = 1
        camK[:, 2, 3] = 1

        self.K = camK

        self.device = device
        lights = AmbientLights(device=self.device)

        raster_settings = RasterizationSettings(
            image_size=output_size,
            blur_radius=0,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardFlatShader(
                device=self.device,
                lights=lights,
                blend_params=BlendParams(background_color=[0, 0, 0], sigma=0.0, gamma=0.0)
            )
        )

    def camera_matrix(self, cam):
        batch_size = cam.size(0)

        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        t = torch.stack(
            [-cam[:, 1], -cam[:, 2], 2 * self.focal_length / (self.orig_size * cam[:, 0] + 1e-9)],
            dim=-1
        )

        if cam.is_cuda:
            # device_id = cam.get_device()
            K = K.to(cam.device)
            R = R.to(cam.device)
            t = t.to(cam.device)

        return K, R, t

    def verts2iuvimg(self, verts, cam, iwp_mode=True):
        batch_size = verts.size(0)

        K, R, t = self.camera_matrix(cam)

        if self.vert_mapping is None:
            vertices = verts
        else:
            vertices = verts[:, self.vert_mapping, :]

        mesh = Meshes(vertices, self.faces.to(verts.device).expand(batch_size, -1, -1))
        mesh.textures = TexturesVertex(
            verts_features=self.textures_vts.to(verts.device).expand(batch_size, -1, -1)
        )

        cameras = PerspectiveCameras(
            device=verts.device,
            R=R,
            T=t,
            K=K,
            in_ndc=False,
            image_size=[(self.orig_size, self.orig_size)]
        )

        iuv_image = self.renderer(mesh, cameras=cameras)
        iuv_image = iuv_image[..., :3].permute(0, 3, 1, 2)

        return iuv_image
