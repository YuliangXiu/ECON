import os
import os.path as osp

import cupy as cp
import cv2
import numpy as np
import torch
import trimesh
from cupyx.scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    hstack,
    spdiags,
    vstack,
)
from cupyx.scipy.sparse.linalg import cg
from PIL import Image
from tqdm.auto import tqdm

from lib.dataset.mesh_util import clean_floats


def find_max_list(lst):
    list_len = [len(i) for i in lst]
    max_id = np.argmax(np.array(list_len))
    return lst[max_id]


def interpolate_pts(pts, diff_ids):

    pts_extend = np.around((pts[diff_ids] + pts[diff_ids - 1]) * 0.5).astype(np.int32)
    pts = np.insert(pts, diff_ids, pts_extend, axis=0)

    return pts


def align_pts(pts1, pts2):

    diff_num = abs(len(pts1) - len(pts2))
    diff_ids = np.sort(np.random.choice(min(len(pts2), len(pts1)), diff_num, replace=True))

    if len(pts1) > len(pts2):
        pts2 = interpolate_pts(pts2, diff_ids)
    elif len(pts2) > len(pts1):
        pts1 = interpolate_pts(pts1, diff_ids)
    else:
        pass

    return pts1, pts2


def repeat_pts(pts1, pts2):

    coverage_mask = ((pts1[:, None, :] == pts2[None, :, :]).sum(axis=2) == 2.).any(axis=1)

    return coverage_mask


def find_contour(mask, method='all'):

    if method == 'all':

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    contour_cloth = np.array(find_max_list(contours))[:, 0, :]

    return contour_cloth


def mean_value_cordinates(inner_pts, contour_pts):

    body_edges_a = np.sqrt(((inner_pts[:, None] - contour_pts[None, :])**2).sum(axis=2))
    body_edges_c = np.roll(body_edges_a, shift=-1, axis=1)
    body_edges_b = np.sqrt(((contour_pts - np.roll(contour_pts, shift=-1, axis=0))**2).sum(axis=1))

    body_edges = np.concatenate([
        body_edges_a[..., None], body_edges_c[..., None],
        np.repeat(body_edges_b[None, :, None], axis=0, repeats=len(inner_pts))
    ],
                                axis=-1)

    body_cos = (body_edges[:, :, 0]**2 + body_edges[:, :, 1]**2 -
                body_edges[:, :, 2]**2) / (2 * body_edges[:, :, 0] * body_edges[:, :, 1])
    body_tan_half = np.sqrt(
        (1. - np.clip(body_cos, a_max=1., a_min=-1.)) / np.clip(1. + body_cos, 1e-6, 2.)
    )

    w = (body_tan_half + np.roll(body_tan_half, shift=1, axis=1)) / body_edges_a
    w /= w.sum(axis=1, keepdims=True)

    return w


def get_dst_mat(contour_body, contour_cloth):

    dst_mat = ((contour_body[:, None, :] - contour_cloth[None, :, :])**2).sum(axis=2)

    return dst_mat


def dispCorres(img_size, contour1, contour2, phi, dir_path):

    contour1 = contour1[None, :, None, :].astype(np.int32)
    contour2 = contour2[None, :, None, :].astype(np.int32)

    disp = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.drawContours(disp, contour1, -1, (0, 255, 0), 1)    # green
    cv2.drawContours(disp, contour2, -1, (255, 0, 0), 1)    # blue

    for i in range(contour1.shape[1]):    # do not show all the points when display
        # cv2.circle(disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), 1,
        #            (255, 0, 0), -1)
        corresPoint = contour2[0, phi[i], 0]
        # cv2.circle(disp, (corresPoint[0], corresPoint[1]), 1, (0, 255, 0), -1)
        cv2.line(
            disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), (corresPoint[0], corresPoint[1]),
            (255, 255, 255), 1
        )

    cv2.imwrite(osp.join(dir_path, "corres.png"), disp)


def remove_stretched_faces(verts, faces):

    mesh = trimesh.Trimesh(verts, faces)
    camera_ray = np.array([0.0, 0.0, 1.0])
    faces_cam_angles = np.dot(mesh.face_normals, camera_ray)

    # cos(90-20)=0.34 cos(90-10)=0.17, 10~20 degree
    faces_mask = np.abs(faces_cam_angles) > 2e-1

    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh.vertices, mesh.faces


def tensor2arr(t, mask=False):
    if not mask:
        return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        mask = t.squeeze(0).abs().sum(dim=0, keepdim=True)
        return (mask != mask[:, 0, 0]).float().squeeze(0).detach().cpu().numpy()


def arr2png(t):
    return ((t + 1.0) * 0.5 * 255.0).astype(np.uint8)


def depth2arr(t):

    return t.float().detach().cpu().numpy()


def depth2png(t):

    t_copy = t.copy()
    t_bg = t_copy[0, 0]
    valid_region = np.logical_and(t > -1.0, t != t_bg)
    t_copy[valid_region] -= t_copy[valid_region].min()
    t_copy[valid_region] /= t_copy[valid_region].max()
    t_copy[valid_region] = (1. - t_copy[valid_region]) * 255.0
    t_copy[~valid_region] = 0.0

    return t_copy[..., None].astype(np.uint8)


def verts_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy *= depth_scale * 0.5
    t_copy += depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]] * torch.Tensor([2.0, 2.0, -2.0]) + torch.Tensor([
        0.0, 0.0, depth_scale
    ])

    return t_copy


def verts_inverse_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy -= torch.tensor([0.0, 0.0, depth_scale])
    t_copy /= torch.tensor([2.0, 2.0, -2.0])
    t_copy -= depth_scale * 0.5
    t_copy /= depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]]

    return t_copy


def depth_inverse_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy -= torch.tensor(depth_scale)
    t_copy /= torch.tensor(-2.0)
    t_copy -= depth_scale * 0.5
    t_copy /= depth_scale * 0.5

    return t_copy


# BNI related


def move_left(mask):
    return cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return cp.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return cp.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return cp.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return cp.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return cp.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def generate_dx_dy_new(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = cp.sum(mask)

    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = cp.stack([-nz_left / step_size, nz_left / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right / step_size, nz_right / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top / step_size, nz_top / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom / step_size, nz_bottom / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = cp.sum(mask)

    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = cp.stack([-nz_left / step_size, nz_left / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right / step_size, nz_right / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top / step_size, nz_top / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom / step_size, nz_bottom / step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]),
                       -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    idx = cp.zeros_like(mask, dtype=int)
    idx[mask] = cp.arange(cp.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)
    facet_top_left_mask = (
        facet_move_top_mask * facet_move_left_mask * facet_move_top_left_mask * mask
    )
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return cp.hstack((
        4 * cp.ones((cp.sum(facet_top_left_mask).item(), 1)),
        idx[facet_top_left_mask][:, None],
        idx[facet_bottom_left_mask][:, None],
        idx[facet_bottom_right_mask][:, None],
        idx[facet_top_right_mask][:, None],
    )).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
    xx = cp.flip(xx, axis=0)

    if K is None:
        vertices = cp.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = cp.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T    # 3 x m
        vertices = (cp.linalg.inv(K) @ u).T * depth_map[mask, cp.newaxis]    # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def boundary_excluded_mask(mask):
    top_mask = cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = cp.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    be_mask = top_mask * bottom_mask * left_mask * right_mask * mask

    # discard single point
    top_mask = cp.pad(be_mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
    bottom_mask = cp.pad(be_mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]
    left_mask = cp.pad(be_mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
    right_mask = cp.pad(be_mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]
    bes_mask = (top_mask + bottom_mask + left_mask + right_mask).astype(bool)
    be_mask = cp.logical_and(be_mask, bes_mask)
    return be_mask


def create_boundary_matrix(mask):
    num_pixel = cp.sum(mask)
    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(num_pixel)

    be_mask = boundary_excluded_mask(mask)
    boundary_mask = cp.logical_xor(be_mask, mask)
    diag_data_term = boundary_mask[mask].astype(int)
    B = diags(diag_data_term)

    num_boundary_pixel = cp.sum(boundary_mask).item()
    data_term = cp.concatenate((cp.ones(num_boundary_pixel), -cp.ones(num_boundary_pixel)))
    row_idx = cp.tile(cp.arange(num_boundary_pixel), 2)
    col_idx = cp.concatenate((pixel_idx[boundary_mask], pixel_idx[boundary_mask] + num_pixel))
    B_full = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_boundary_pixel, 2 * num_pixel))
    return B, B_full


def double_side_bilateral_normal_integration(
    normal_front,
    normal_back,
    normal_mask,
    depth_front=None,
    depth_back=None,
    depth_mask=None,
    k=2,
    lambda_normal_back=1,
    lambda_depth_front=1e-4,
    lambda_depth_back=1e-2,
    lambda_boundary_consistency=1,
    step_size=1,
    max_iter=150,
    tol=1e-4,
    cg_max_iter=5000,
    cg_tol=1e-3,
    cut_intersection=True,
):

    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]

    num_normals = cp.sum(normal_mask)
    normal_map_front = cp.asarray(normal_front)
    normal_map_back = cp.asarray(normal_back)
    normal_mask = cp.asarray(normal_mask)
    if depth_mask is not None:
        depth_map_front = cp.asarray(depth_front)
        depth_map_back = cp.asarray(depth_back)
        depth_mask = cp.asarray(depth_mask)

    # transfer the normal map from the normal coordinates to the camera coordinates
    nx_front = normal_map_front[normal_mask, 1]
    ny_front = normal_map_front[normal_mask, 0]
    nz_front = -normal_map_front[normal_mask, 2]
    del normal_map_front

    nx_back = normal_map_back[normal_mask, 1]
    ny_back = normal_map_back[normal_mask, 0]
    nz_back = -normal_map_back[normal_mask, 2]
    del normal_map_back

    # right, left, top, bottom
    A3_f, A4_f, A1_f, A2_f = generate_dx_dy(
        normal_mask, nz_horizontal=nz_front, nz_vertical=nz_front, step_size=step_size
    )
    A3_b, A4_b, A1_b, A2_b = generate_dx_dy(
        normal_mask, nz_horizontal=nz_back, nz_vertical=nz_back, step_size=step_size
    )

    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)

    top_boundnary_mask = cp.logical_xor(has_top_mask, normal_mask)[normal_mask]
    bottom_boundary_mask = cp.logical_xor(has_bottom_mask, normal_mask)[normal_mask]
    left_boundary_mask = cp.logical_xor(has_left_mask, normal_mask)[normal_mask]
    right_boudnary_mask = cp.logical_xor(has_right_mask, normal_mask)[normal_mask]

    A_front_data = vstack((A1_f, A2_f, A3_f, A4_f))
    A_front_zero = csr_matrix(A_front_data.shape)
    A_front = hstack([A_front_data, A_front_zero])

    A_back_data = vstack((A1_b, A2_b, A3_b, A4_b))
    A_back_zero = csr_matrix(A_back_data.shape)
    A_back = hstack([A_back_zero, A_back_data])

    b_front = cp.concatenate((-nx_front, -nx_front, -ny_front, -ny_front))
    b_back = cp.concatenate((-nx_back, -nx_back, -ny_back, -ny_back))

    # initialization
    W_front = spdiags(
        0.5 * cp.ones(4 * num_normals), 0, 4 * num_normals, 4 * num_normals, format="csr"
    )
    W_back = spdiags(
        0.5 * cp.ones(4 * num_normals), 0, 4 * num_normals, 4 * num_normals, format="csr"
    )

    z_front = cp.zeros(num_normals, float)
    z_back = cp.zeros(num_normals, float)
    z_combined = cp.concatenate((z_front, z_back))

    B, B_full = create_boundary_matrix(normal_mask)
    B_mat = lambda_boundary_consistency * coo_matrix(B_full.get().T @ B_full.get())    #bug

    energy_list = []

    if depth_mask is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)    # shape: (num_normals,)
        z_prior_front = depth_map_front[normal_mask]    # shape: (num_normals,)
        z_prior_front[~depth_mask_flat] = 0
        z_prior_back = depth_map_back[normal_mask]
        z_prior_back[~depth_mask_flat] = 0
        m = depth_mask[normal_mask].astype(int)
        M = diags(m)

    energy = (A_front @ z_combined - b_front).T @ W_front @ (A_front @ z_combined - b_front) + \
             lambda_normal_back * (A_back @ z_combined - b_back).T @ W_back @ (A_back @ z_combined - b_back) + \
             lambda_depth_front * (z_front - z_prior_front).T @ M @ (z_front - z_prior_front) + \
             lambda_depth_back * (z_back - z_prior_back).T @ M @ (z_back - z_prior_back) + \
             lambda_boundary_consistency * (z_back - z_front).T @ B @ (z_back - z_front)

    depth_map_front_est = cp.ones_like(normal_mask, float) * cp.nan
    depth_map_back_est = cp.ones_like(normal_mask, float) * cp.nan

    facets_back = cp.asnumpy(construct_facets_from(normal_mask))
    faces_back = np.concatenate((facets_back[:, [1, 4, 3]], facets_back[:, [1, 3, 2]]), axis=0)
    faces_front = np.concatenate((facets_back[:, [1, 2, 3]], facets_back[:, [1, 3, 4]]), axis=0)

    for i in range(max_iter):
        A_mat_front = A_front_data.T @ W_front @ A_front_data
        b_vec_front = A_front_data.T @ W_front @ b_front

        A_mat_back = A_back_data.T @ W_back @ A_back_data
        b_vec_back = A_back_data.T @ W_back @ b_back
        if depth_mask is not None:
            b_vec_front += lambda_depth_front * M @ z_prior_front
            b_vec_back += lambda_depth_back * M @ z_prior_back
            A_mat_front += lambda_depth_front * M
            A_mat_back += lambda_depth_back * M
            offset_front = cp.mean((z_prior_front - z_combined[:num_normals])[depth_mask_flat])
            offset_back = cp.mean((z_prior_back - z_combined[num_normals:])[depth_mask_flat])
            z_combined[:num_normals] = z_combined[:num_normals] + offset_front
            z_combined[num_normals:] = z_combined[num_normals:] + offset_back


        A_mat_combined = hstack([vstack((A_mat_front, csr_matrix((num_normals, num_normals)))), \
                                 vstack((csr_matrix((num_normals, num_normals)), A_mat_back))]) + B_mat
        b_vec_combined = cp.concatenate((b_vec_front, b_vec_back))

        D = spdiags(
            1 / cp.clip(A_mat_combined.diagonal(), 1e-5, None), 0, 2 * num_normals, 2 * num_normals,
            "csr"
        )    # Jacob preconditioner

        z_combined, _ = cg(
            A_mat_combined, b_vec_combined, M=D, x0=z_combined, maxiter=cg_max_iter, tol=cg_tol
        )
        z_front = z_combined[:num_normals]
        z_back = z_combined[num_normals:]
        wu_f = sigmoid((A2_f.dot(z_front))**2 - (A1_f.dot(z_front))**2, k)    # top
        wv_f = sigmoid((A4_f.dot(z_front))**2 - (A3_f.dot(z_front))**2, k)    # right
        wu_f[top_boundnary_mask] = 0.5
        wu_f[bottom_boundary_mask] = 0.5
        wv_f[left_boundary_mask] = 0.5
        wv_f[right_boudnary_mask] = 0.5
        W_front = spdiags(
            cp.concatenate((wu_f, 1 - wu_f, wv_f, 1 - wv_f)),
            0,
            4 * num_normals,
            4 * num_normals,
            format="csr"
        )

        wu_b = sigmoid((A2_b.dot(z_back))**2 - (A1_b.dot(z_back))**2, k)    # top
        wv_b = sigmoid((A4_b.dot(z_back))**2 - (A3_b.dot(z_back))**2, k)    # right
        wu_b[top_boundnary_mask] = 0.5
        wu_b[bottom_boundary_mask] = 0.5
        wv_b[left_boundary_mask] = 0.5
        wv_b[right_boudnary_mask] = 0.5
        W_back = spdiags(
            cp.concatenate((wu_b, 1 - wu_b, wv_b, 1 - wv_b)),
            0,
            4 * num_normals,
            4 * num_normals,
            format="csr"
        )

        energy_old = energy
        energy = (A_front_data @ z_front - b_front).T @ W_front @ (A_front_data @ z_front - b_front) + \
             lambda_normal_back * (A_back_data @ z_back - b_back).T @ W_back @ (A_back_data @ z_back - b_back) + \
             lambda_depth_front * (z_front - z_prior_front).T @ M @ (z_front - z_prior_front) + \
             lambda_depth_back * (z_back - z_prior_back).T @ M @ (z_back - z_prior_back) +\
             lambda_boundary_consistency * (z_back - z_front).T @ B @ (z_back - z_front)

        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old

        # print(f"step {i + 1}/{max_iter} energy: {energy:.3e}"
        #       f" relative energy: {relative_energy:.3e}")

        if False:
            # intermediate results
            depth_map_front_est[normal_mask] = z_front
            depth_map_back_est[normal_mask] = z_back
            vertices_front = cp.asnumpy(
                map_depth_map_to_point_clouds(
                    depth_map_front_est, normal_mask, K=None, step_size=step_size
                )
            )
            vertices_back = cp.asnumpy(
                map_depth_map_to_point_clouds(
                    depth_map_back_est, normal_mask, K=None, step_size=step_size
                )
            )

            vertices_front, faces_front_ = remove_stretched_faces(vertices_front, faces_front)
            vertices_back, faces_back_ = remove_stretched_faces(vertices_back, faces_back)

            F_verts = verts_inverse_transform(torch.as_tensor(vertices_front).float(), 256.0)
            B_verts = verts_inverse_transform(torch.as_tensor(vertices_back).float(), 256.0)

            F_B_verts = torch.cat((F_verts, B_verts), dim=0)
            F_B_faces = torch.cat((
                torch.as_tensor(faces_front_).long(),
                torch.as_tensor(faces_back_).long() + faces_front_.max() + 1
            ),
                                  dim=0)

            front_surf = trimesh.Trimesh(F_verts, faces_front_)
            back_surf = trimesh.Trimesh(B_verts, faces_back_)
            double_surf = trimesh.Trimesh(F_B_verts, F_B_faces)

            bini_dir = "/home/yxiu/Code/ECON/log/bini/OBJ"
            front_surf.export(osp.join(bini_dir, f"{i:04d}_F.obj"))
            back_surf.export(osp.join(bini_dir, f"{i:04d}_B.obj"))
            double_surf.export(osp.join(bini_dir, f"{i:04d}_FB.obj"))

        if relative_energy < tol:
            break
    # del A1, A2, A3, A4, nx, ny

    depth_map_front_est[normal_mask] = z_front
    depth_map_back_est[normal_mask] = z_back

    if cut_intersection:
        # manually cut the intersection
        normal_mask[depth_map_front_est >= depth_map_back_est] = False
        depth_map_front_est[~normal_mask] = cp.nan
        depth_map_back_est[~normal_mask] = cp.nan

    vertices_front = cp.asnumpy(
        map_depth_map_to_point_clouds(
            depth_map_front_est, normal_mask, K=None, step_size=step_size
        )
    )
    vertices_back = cp.asnumpy(
        map_depth_map_to_point_clouds(depth_map_back_est, normal_mask, K=None, step_size=step_size)
    )

    facets_back = cp.asnumpy(construct_facets_from(normal_mask))
    faces_back = np.concatenate((facets_back[:, [1, 4, 3]], facets_back[:, [1, 3, 2]]), axis=0)
    faces_front = np.concatenate((facets_back[:, [1, 2, 3]], facets_back[:, [1, 3, 4]]), axis=0)

    vertices_front, faces_front = remove_stretched_faces(vertices_front, faces_front)
    vertices_back, faces_back = remove_stretched_faces(vertices_back, faces_back)

    front_mesh = clean_floats(trimesh.Trimesh(vertices_front, faces_front))
    back_mesh = clean_floats(trimesh.Trimesh(vertices_back, faces_back))

    result = {
        "F_verts": torch.as_tensor(front_mesh.vertices).float(), "F_faces": torch.as_tensor(
            front_mesh.faces
        ).long(), "B_verts": torch.as_tensor(back_mesh.vertices).float(), "B_faces":
        torch.as_tensor(back_mesh.faces).long(), "F_depth":
        torch.as_tensor(depth_map_front_est).float(), "B_depth":
        torch.as_tensor(depth_map_back_est).float()
    }

    return result


def save_normal_tensor(in_tensor, idx, png_path, thickness=0.0):

    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    normal_F_arr = tensor2arr(in_tensor["normal_F"][idx:idx + 1])
    normal_B_arr = tensor2arr(in_tensor["normal_B"][idx:idx + 1])
    mask_normal_arr = tensor2arr(in_tensor["image"][idx:idx + 1], True)

    depth_F_arr = depth2arr(in_tensor["depth_F"][idx])
    depth_B_arr = depth2arr(in_tensor["depth_B"][idx])

    BNI_dict = {}

    # clothed human
    BNI_dict["normal_F"] = normal_F_arr
    BNI_dict["normal_B"] = normal_B_arr
    BNI_dict["mask"] = mask_normal_arr > 0.
    BNI_dict["depth_F"] = depth_F_arr - 100. - thickness
    BNI_dict["depth_B"] = 100. - depth_B_arr + thickness
    BNI_dict["depth_mask"] = depth_F_arr != -1.0

    np.save(png_path + ".npy", BNI_dict, allow_pickle=True)

    return BNI_dict
