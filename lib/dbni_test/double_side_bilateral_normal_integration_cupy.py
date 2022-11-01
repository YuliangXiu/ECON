"""
Double-side depth-aware Bilateral Normal Integration (d^2-BiNI)
"""
__author__ = "Xu Cao <cao.xu@ist.osaka-u.ac.jp>"
__copyright__ = "Copyright (C) 2022 Xu Cao"
__version__ = "2.1"

import pyvista
import pyvista as pv
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix, vstack, hstack, spdiags, diags, coo_matrix
from cupyx.scipy.sparse.linalg import cg
from tqdm.auto import tqdm
import time

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


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

    data = cp.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def boundary_excluded_mask(mask):
    top_mask =cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
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
    col_idx = cp.concatenate((pixel_idx[boundary_mask], pixel_idx[boundary_mask]+num_pixel))
    B_full = coo_matrix(
         (data_term, (row_idx, col_idx)), shape=(num_boundary_pixel, 2 * num_pixel))
    return B, B_full


def construct_facets_from(mask):
    idx = cp.zeros_like(mask, dtype=int)
    idx[mask] = cp.arange(cp.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)
    facet_top_left_mask = facet_move_top_mask * facet_move_left_mask * facet_move_top_left_mask * mask
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return cp.stack((4 * cp.ones(cp.sum(facet_top_left_mask).item()),
                      idx[facet_top_left_mask],
                      idx[facet_bottom_left_mask],
                      idx[facet_bottom_right_mask],
                      idx[facet_top_right_mask]), axis=-1).astype(int)


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
        u = u[mask].T  # 3 x m
        vertices = (cp.linalg.inv(cp.asarray(K)) @ u).T * \
            depth_map[mask, cp.newaxis]  # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def double_side_bilateral_normal_integration(normal_front,
                                 normal_back,
                                 normal_mask,
                                 depth_front=None,
                                 depth_back=None,
                                 depth_mask=None,
                                 k=2,
                                 lambda_normal_back=1,
                                 lambda_depth_front = 1e-4,
                                 lambda_depth_back = 1e-2,
                                 lambda_boundary_consistency=1,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3):

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
    num_normals = np.sum(normal_mask)
    normal_map_front = cp.asarray(normal_front)
    normal_map_back = cp.asarray(normal_back)
    normal_mask = cp.asarray(normal_mask)
    if depth_mask is not None:
        depth_map_front = cp.asarray(depth_front)
        depth_map_back = cp.asarray(depth_back)
        depth_mask = cp.asarray(depth_mask)

    # num_normals = cp.sum(normal_mask).item()
    print(f"Running bilateral normal integration with k={k} in the orthographic case. \n"
          f"The number of normal vectors is {num_normals}.")
    # transfer the normal map from the normal coordinates to the camera coordinates
    nx_front = normal_map_front[normal_mask, 1]
    ny_front = normal_map_front[normal_mask, 0]
    nz_front = - normal_map_front[normal_mask, 2]
    del normal_map_front

    nx_back = normal_map_back[normal_mask, 1]
    ny_back = normal_map_back[normal_mask, 0]
    nz_back = - normal_map_back[normal_mask, 2]
    del normal_map_back

    # right, left, top, bottom
    A3_f, A4_f, A1_f, A2_f = generate_dx_dy(normal_mask, nz_horizontal=nz_front, nz_vertical=nz_front, step_size=step_size)
    A3_b, A4_b, A1_b, A2_b = generate_dx_dy(normal_mask, nz_horizontal=nz_back, nz_vertical=nz_back, step_size=step_size)

    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)

    top_boundnary_mask = cp.logical_xor(has_top_mask, normal_mask)[normal_mask]
    bottom_boundary_mask = cp.logical_xor(has_bottom_mask, normal_mask)[normal_mask]
    left_boundary_mask = cp.logical_xor(has_left_mask, normal_mask)[normal_mask]
    right_boudnary_mask = cp.logical_xor(has_right_mask, normal_mask)[normal_mask]


    A_front_data = vstack((A1_f, A2_f, A3_f, A4_f))
    # print(A_front_data.shape)
    A_front_zero = csr_matrix(A_front_data.shape)
    A_front = hstack([A_front_data, A_front_zero])
    # print(A_front.shape)
    # print(A_front.nnz, A_front_data.nnz, A_front_zero.nnz)

    A_back_data = vstack((A1_b, A2_b, A3_b, A4_b))
    A_back_zero = csr_matrix(A_back_data.shape)
    A_back = hstack([A_back_zero, A_back_data])

    b_front = cp.concatenate((-nx_front, -nx_front, -ny_front, -ny_front))
    b_back = cp.concatenate((-nx_back, -nx_back, -ny_back, -ny_back))

    # initialization
    W_front = spdiags(0.5 * np.ones(4 * num_normals), 0, 4 * num_normals, 4 * num_normals, format="csr")
    W_back = spdiags(0.5 * np.ones(4 * num_normals), 0, 4 * num_normals, 4 * num_normals, format="csr")
    z_front = cp.zeros(num_normals, float)
    z_back = cp.zeros(num_normals, float)
    z_combined = cp.concatenate((z_front, z_back))

    B, B_full = create_boundary_matrix(normal_mask)
    B_mat = lambda_boundary_consistency * B_full.T @ B_full

    tic = time.time()

    energy_list = []

    if depth_mask is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)  # shape: (num_normals,)
        z_prior_front = depth_map_front[normal_mask]  # shape: (num_normals,)
        z_prior_front[~depth_mask_flat] = 0
        z_prior_back = depth_map_back[normal_mask]
        z_prior_back[~depth_mask_flat] = 0
        m = depth_mask[normal_mask].astype(int)
        M = diags(m)
        # z_prior_combined = cp.concatenate((lambda_depth_front * z_prior_front, lambda_depth_back * z_prior_back))

    energy = (A_front @ z_combined - b_front).T @ W_front @ (A_front @ z_combined - b_front) + \
             lambda_normal_back * (A_back @ z_combined - b_back).T @ W_back @ (A_back @ z_combined - b_back) + \
             lambda_depth_front * (z_front - z_prior_front).T @ M @ (z_front - z_prior_front) + \
             lambda_depth_back * (z_back - z_prior_back).T @ M @ (z_back - z_prior_back) + \
             lambda_boundary_consistency * (z_back - z_front).T @ B @ (z_back - z_front)

    pbar = tqdm(range(max_iter))

    for i in pbar:
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
            # z_front = z_front + offset_front
            # z_back = z_back + offset_back

        A_mat_combined = hstack([vstack((A_mat_front, csr_matrix((num_normals, num_normals)))), \
                                 vstack((csr_matrix((num_normals, num_normals)), A_mat_back))]) + B_mat
        b_vec_combined = cp.concatenate((b_vec_front, b_vec_back))

        # D_front = spdiags(1/cp.clip(A_mat_front.diagonal(), 1e-5, None), 0, num_normals, num_normals, "csr")  # Jacob preconditioner
        # D_back = spdiags(1/cp.clip(A_mat_back.diagonal(), 1e-5, None), 0, num_normals, num_normals, "csr")
        D = spdiags(1/cp.clip(A_mat_combined.diagonal(), 1e-5, None), 0, 2*num_normals, 2*num_normals, "csr")  # Jacob preconditioner

        # z_front, _ = cg(A_mat_front, b_vec_front, x0=z_front, M=D_front, maxiter=cg_max_iter, tol=cg_tol)
        # z_back, _ = cg(A_mat_back, b_vec_back, x0=z_back, M=D_back, maxiter=cg_max_iter, tol=cg_tol)
        z_combined, _ = cg(A_mat_combined, b_vec_combined, M=D, x0=z_combined, maxiter=cg_max_iter, tol=cg_tol)
        z_front = z_combined[:num_normals]
        z_back = z_combined[num_normals:]
        # update weights
        # wu_f = sigmoid((A2_f.dot(z_comnined[:num_normals])) ** 2 - (A1_f.dot(z_comnined[:num_normals])) ** 2, k)  # top
        # wv_f = sigmoid((A4_f.dot(z_comnined[:num_normals])) ** 2 - (A3_f.dot(z_comnined[:num_normals])) ** 2, k)  # right
        wu_f = sigmoid((A2_f.dot(z_front)) ** 2 - (A1_f.dot(z_front)) ** 2, k)  # top
        wv_f = sigmoid((A4_f.dot(z_front)) ** 2 - (A3_f.dot(z_front)) ** 2, k)  # right
        wu_f[top_boundnary_mask] = 0.5
        wu_f[bottom_boundary_mask] = 0.5
        wv_f[left_boundary_mask] = 0.5
        wv_f[right_boudnary_mask] = 0.5
        W_front = spdiags(np.concatenate((wu_f, 1-wu_f, wv_f, 1-wv_f)), 0, 4*num_normals, 4*num_normals, format="csr")

        wu_b = sigmoid((A2_b.dot(z_back)) ** 2 - (A1_b.dot(z_back)) ** 2, k)  # top
        wv_b = sigmoid((A4_b.dot(z_back)) ** 2 - (A3_b.dot(z_back)) ** 2, k)  # right
        wu_b[top_boundnary_mask] = 0.5
        wu_b[bottom_boundary_mask] = 0.5
        wv_b[left_boundary_mask] = 0.5
        wv_b[right_boudnary_mask] = 0.5
        W_back = spdiags(np.concatenate((wu_b, 1 - wu_b, wv_b, 1 - wv_b)), 0, 4 * num_normals, 4 * num_normals,
                          format="csr")

        # wu_b = sigmoid((A2_b.dot(z_comnined[num_normals:])) ** 2 - (A1_b.dot(z_comnined[num_normals:])) ** 2, k)  # top
        # wv_b = sigmoid((A4_b.dot(z_comnined[num_normals:])) ** 2 - (A3_b.dot(z_comnined[num_normals:])) ** 2, k)  # right
        # W_back = spdiags(np.concatenate((wu_b, 1-wu_b, wv_b, 1-wv_b)), 0, 4*num_normals, 4*num_normals, format="csr")

        energy_old = energy
        energy = (A_front_data @ z_front - b_front).T @ W_front @ (A_front_data @ z_front - b_front) + \
             lambda_normal_back * (A_back_data @ z_back - b_back).T @ W_back @ (A_back_data @ z_back - b_back) + \
             lambda_depth_front * (z_front - z_prior_front).T @ M @ (z_front - z_prior_front) + \
             lambda_depth_back * (z_back - z_prior_back).T @ M @ (z_back - z_prior_back) +\
             lambda_boundary_consistency * (z_back - z_front).T @ B @ (z_back - z_front)

        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3e}"
            f" relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    # del A1, A2, A3, A4, nx, ny
    toc = time.time()

    print(f"Total time: {toc - tic:.3f} sec")
    depth_map_front_est = cp.ones_like(normal_mask, float) * cp.nan
    # depth_map_front_est[normal_mask] = z_combined[:num_normals]
    depth_map_front_est[normal_mask] = z_front

    depth_map_back_est = cp.ones_like(normal_mask, float) * cp.nan
    # depth_map_back_est[normal_mask] = z_combined[num_normals:]
    depth_map_back_est[normal_mask] = z_back

    vertices_front = cp.asnumpy(map_depth_map_to_point_clouds(depth_map_front_est, normal_mask, K=None, step_size=step_size))
    vertices_back = cp.asnumpy(map_depth_map_to_point_clouds(depth_map_back_est, normal_mask, K=None, step_size=step_size))

    facets_back = cp.asnumpy(construct_facets_from(normal_mask))
    facets_front = facets_back.copy()[:, [0, 1, 4, 3, 2]]

    surface_front = pv.PolyData(vertices_front, facets_front)
    surface_back = pv.PolyData(vertices_back, facets_back)


    # # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    # wu_map = cp.ones_like(normal_mask) * cp.nan
    # wu_map[normal_mask] = wv
    #
    # wv_map = cp.ones_like(normal_mask) * cp.nan
    # wv_map[normal_mask] = wu
    #
    # depth_map = cp.asnumpy(depth_map)
    # wu_map = cp.asnumpy(wu_map)
    # wv_map = cp.asnumpy(wv_map)
    
    return surface_front, surface_back, depth_map_front_est, depth_map_back_est
    # return surface_front, depth_map_front_est


if __name__ == '__main__':
    import cv2
    import argparse
    import os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + cp.exp(-k * x)).
    # This overflow issue does not affect our results as cp.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=int, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    parser.add_argument('--cgiter', type=int, default=5000)
    parser.add_argument('--cgtol', type=float, default=1e-3)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(
        arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)

    if os.path.exists(os.path.join(arg.path, "K.txt")):
        K = np.loadtxt(os.path.join(arg.path, "K.txt"))
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=K,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol,
                                                                                       cg_max_iter=arg.cgiter,
                                                                                       cg_tol=arg.cgtol)
    else:
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=None,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol,
                                                                                       cg_max_iter=arg.cgiter,
                                                                                       cg_tol=arg.cgtol)

    # save the resultant polygon mesh and discontinuity maps.
    cp.save(os.path.join(arg.path, "energy"), cp.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap(
        (255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap(
        (255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"saved {arg.path}")
