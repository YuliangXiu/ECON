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

import numbers

import numpy as np
import torch
from einops.einops import rearrange
from torch.nn import functional as F

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2,
        2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ],
                         dim=1).view(B, 3, 3)
    return rotMat


def index(feat, uv):
    """
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    uv = uv.transpose(1, 2)    # [B, N, 2]

    (B, N, _) = uv.shape
    C = feat.shape[1]

    if uv.shape[-1] == 3:
        # uv = uv[:,:,[2,1,0]]
        # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
        uv = uv.unsqueeze(2).unsqueeze(3)    # [B, N, 1, 1, 3]
    else:
        uv = uv.unsqueeze(2)    # [B, N, 1, 2]

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)    # [B, C, N, 1]
    return samples.view(B, C, N)    # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx3x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(
            1, 3, 1
        ).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(rotation_matrix.shape)
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(rotation_matrix.shape)
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2    # noqa
        + t3_rep * mask_c3
    )    # noqa
    q *= 0.5
    return q


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    if x.shape[-1] == 6:
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            num = x.shape[1]
            x = rearrange(x, 'b n d -> (b n) d', d=6)
        else:
            num = 1
        x = rearrange(x, 'b (k l) -> b k l', k=3, l=2)
        # x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)

        mat = torch.stack((b1, b2, b3), dim=-1)
        if num > 1:
            mat = rearrange(mat, '(b n) h w-> b n h w', b=batch_size, n=num, h=3, w=3)
    else:
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        mat = torch.stack((b1, b2, b3), dim=-1)
    return mat


def rotmat_to_rot6d(x):
    """Convert 3x3 rotation matrix to 6D rotation representation.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,3,3) Batch of corresponding rotation matrices
    Output:
        (B,6) Batch of 6-D rotation representations
    """
    batch_size = x.shape[0]
    x = x[:, :, :2]
    x = x.reshape(batch_size, 6)
    return x


def rotmat_to_angle(x):
    """Convert rotation to one-D angle.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,2) Batch of corresponding rotation
    Output:
        (B,1) Batch of 1-D angle
    """
    a = F.normalize(x)
    angle = torch.atan2(a[:, 0], a[:, 1]).unsqueeze(-1)

    return angle


def projection(pred_joints, pred_camera, retain_z=False):
    pred_cam_t = torch.stack(
        [
            pred_camera[:, 1],
            pred_camera[:, 2],
            2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(
        pred_joints,
        rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
        translation=pred_cam_t,
        focal_length=5000.0,
        camera_center=camera_center,
        retain_z=retain_z,
    )
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


def perspective_projection(
    points, rotation, translation, focal_length, camera_center, retain_z=False
):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    if retain_z:
        return projected_points
    else:
        return projected_points[:, :, :-1]


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=(224., 224.)):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
    center = np.array([img_size[1] / 2., img_size[0] / 2.])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([
        F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
        O - np.reshape(joints_2d, -1)
    ]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224., use_all_kps=False):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if isinstance(focal_length, numbers.Number):
        focal_length = [
            focal_length,
        ] * S.shape[0]
        # print(len(focal_length), focal_length)

    if isinstance(img_size, numbers.Number):
        img_size = [
            (img_size, img_size),
        ] * S.shape[0]
        # print(len(img_size), img_size)

    device = S.device
    if use_all_kps:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        # Use only joints 25:49 (GT joints)
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length[i], img_size=img_size[i]
        )
    return torch.from_numpy(trans).to(device)


def Rot_y(angle, category="torch", prepend_dim=True, device=None):
    """Rotate around y-axis by angle
    Args:
            category: 'torch' or 'numpy'
            prepend_dim: prepend an extra dimension
    Return: Rotation matrix with shape [1, 3, 3] (prepend_dim=True)
    """
    m = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    if category == "torch":
        if prepend_dim:
            return torch.tensor(m, dtype=torch.float, device=device).unsqueeze(0)
        else:
            return torch.tensor(m, dtype=torch.float, device=device)
    elif category == "numpy":
        if prepend_dim:
            return np.expand_dims(m, 0)
        else:
            return m
    else:
        raise ValueError("category must be 'torch' or 'numpy'")


def Rot_x(angle, category="torch", prepend_dim=True, device=None):
    """Rotate around x-axis by angle
    Args:
            category: 'torch' or 'numpy'
            prepend_dim: prepend an extra dimension
    Return: Rotation matrix with shape [1, 3, 3] (prepend_dim=True)
    """
    m = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])
    if category == "torch":
        if prepend_dim:
            return torch.tensor(m, dtype=torch.float, device=device).unsqueeze(0)
        else:
            return torch.tensor(m, dtype=torch.float, device=device)
    elif category == "numpy":
        if prepend_dim:
            return np.expand_dims(m, 0)
        else:
            return m
    else:
        raise ValueError("category must be 'torch' or 'numpy'")


def Rot_z(angle, category="torch", prepend_dim=True, device=None):
    """Rotate around z-axis by angle
    Args:
            category: 'torch' or 'numpy'
            prepend_dim: prepend an extra dimension
    Return: Rotation matrix with shape [1, 3, 3] (prepend_dim=True)
    """
    m = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    if category == "torch":
        if prepend_dim:
            return torch.tensor(m, dtype=torch.float, device=device).unsqueeze(0)
        else:
            return torch.tensor(m, dtype=torch.float, device=device)
    elif category == "numpy":
        if prepend_dim:
            return np.expand_dims(m, 0)
        else:
            return m
    else:
        raise ValueError("category must be 'torch' or 'numpy'")


def compute_twist_rotation(rotation_matrix, twist_axis):
    '''
    Compute the twist component of given rotation and twist axis
    https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
    Parameters
    ----------
    rotation_matrix : Tensor (B, 3, 3,)
        The rotation to convert
    twist_axis : Tensor (B, 3,)
        The twist axis
    Returns
    -------
    Tensor (B, 3, 3)
        The twist rotation
    '''
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    twist_axis = twist_axis / (torch.norm(twist_axis, dim=1, keepdim=True) + 1e-9)

    projection = torch.einsum('bi,bi->b', twist_axis, quaternion[:, 1:]).unsqueeze(-1) * twist_axis

    twist_quaternion = torch.cat([quaternion[:, 0:1], projection], dim=1)
    twist_quaternion = twist_quaternion / (torch.norm(twist_quaternion, dim=1, keepdim=True) + 1e-9)

    twist_rotation = quaternion_to_rotation_matrix(twist_quaternion)
    twist_aa = quaternion_to_angle_axis(twist_quaternion)

    twist_angle = torch.sum(twist_aa, dim=1,
                            keepdim=True) / torch.sum(twist_axis, dim=1, keepdim=True)

    return twist_rotation, twist_angle
