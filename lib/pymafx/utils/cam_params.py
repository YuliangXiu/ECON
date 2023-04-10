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

import os

import joblib
import numpy as np
import torch
from numpy.testing._private.utils import print_assert_equal

from .geometry import batch_euler2matrix


def f_pix2vfov(f_pix, img_h):

    if torch.is_tensor(f_pix):
        fov = 2. * torch.arctan(img_h / (2. * f_pix))
    else:
        fov = 2. * np.arctan(img_h / (2. * f_pix))

    return fov


def vfov2f_pix(fov, img_h):

    if torch.is_tensor(fov):
        f_pix = img_h / 2. / torch.tan(fov / 2.)
    else:
        f_pix = img_h / 2. / np.tan(fov / 2.)

    return f_pix


def read_cam_params(cam_params, orig_shape=None):
    # These are predicted camera parameters
    # cam_param_folder = CAM_PARAM_FOLDERS[dataset_name][cam_param_type]

    cam_pitch = cam_params['pitch'].item()
    cam_roll = cam_params['roll'].item() if 'roll' in cam_params else None

    cam_vfov = cam_params['vfov'].item() if 'vfov' in cam_params else None

    cam_focal_length = cam_params['f_pix']

    orig_shape = cam_params['orig_resolution']

    # cam_rotmat = batch_euler2matrix(torch.tensor([[cam_pitch, 0., cam_roll]]).float())[0]
    cam_rotmat = batch_euler2matrix(torch.tensor([[cam_pitch, 0., 0.]]).float())[0]

    pred_cam_int = torch.zeros(3, 3)

    cx, cy = orig_shape[1] / 2, orig_shape[0] / 2

    pred_cam_int[0, 0] = cam_focal_length
    pred_cam_int[1, 1] = cam_focal_length

    pred_cam_int[:-1, -1] = torch.tensor([cx, cy])

    cam_int = pred_cam_int.float()

    return cam_rotmat, cam_int, cam_vfov, cam_pitch, cam_roll, cam_focal_length


def homo_vector(vector):
    """
    vector: B x N x C
    h_vector: B x N x (C + 1)
    """

    batch_size, n_pts = vector.shape[:2]

    h_vector = torch.cat([vector, torch.ones((batch_size, n_pts, 1)).to(vector)], dim=-1)
    return h_vector
