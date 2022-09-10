import torch
import trimesh
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import diags, coo_matrix, vstack
from cupyx.scipy.sparse.linalg import cg
from tqdm.auto import tqdm


def remove_stretched_faces(verts, faces):

    mesh = trimesh.Trimesh(verts, faces)
    camera_ray = np.array([0.0, 0.0, 1.0])
    faces_cam_angles = np.dot(mesh.face_normals, camera_ray)

    # cos(90-20)=0.34 cos(90-10)=0.17, 10~20 degree
    faces_mask = np.abs(faces_cam_angles) > 3e-1

    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh.vertices, mesh.faces


def tensor2arr(t, mask=False):
    if not mask:
        return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        return ((t.squeeze(0).abs().sum(dim=0, keepdim=True) !=
                 0.0).float().permute(1, 2, 0).detach().cpu().numpy())


def arr2png(t):
    return ((t + 1.0) * 0.5 * 255.0).astype(np.uint8)


def depth2arr(t):

    return t.float().detach().cpu().numpy()


def depth2png(t):

    t_copy = t.copy()
    t_copy -= t_copy[t > -1.0].min()
    t_copy /= t_copy[t > -1.0].max()
    t_copy = (-t_copy + 1.0) * 255.0

    return t_copy[..., None].astype(np.uint8)


def verts_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy *= depth_scale * 0.5
    t_copy += depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]] * torch.Tensor(
        [2.0, 2.0, -2.0]) + torch.Tensor([0.0, 0.0, depth_scale])

    return t_copy


def verts_inverse_transform(t, depth_scale):

    t_copy = t.clone()
    t_copy -= torch.tensor([0.0, 0.0, depth_scale])
    t_copy /= torch.tensor([2.0, 2.0, -2.0])
    t_copy -= depth_scale * 0.5
    t_copy /= depth_scale * 0.5
    t_copy = t_copy[:, [1, 0, 2]]

    return t_copy


# BNI related


def move_left(mask):
    return cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return cp.pad(mask, ((0, 0), (1, 0)), "constant",
                  constant_values=0)[:, :-1]


def move_top(mask):
    return cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return cp.pad(mask, ((1, 0), (0, 0)), "constant",
                  constant_values=0)[:-1, :]


def move_top_left(mask):
    return cp.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:,
                                                                         1:]


def move_top_right(mask):
    return cp.pad(mask, ((0, 1), (1, 0)), "constant",
                  constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return cp.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1,
                                                                         1:]


def move_bottom_right(mask):
    return cp.pad(mask, ((1, 0), (1, 0)), "constant",
                  constant_values=0)[:-1, :-1]


def generate_dx_dy(mask, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive

    pixel_idx = cp.zeros_like(mask, dtype=int)
    pixel_idx[mask] = cp.arange(cp.sum(mask))
    num_pixel = cp.sum(mask)

    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    data_term = cp.array([-1] * int(cp.sum(has_left_mask)) +
                         [1] * int(cp.sum(has_left_mask))).astype(cp.float32)

    # only the pixels having left neighbors have [-1, 1] in that row
    row_idx = pixel_idx[has_left_mask]
    row_idx = cp.tile(row_idx, 2)
    col_idx = cp.concatenate(
        (pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]))
    D_horizontal_neg = coo_matrix((data_term, (row_idx, col_idx)),
                                  shape=(num_pixel, num_pixel))

    data_term = cp.array([-1] * int(cp.sum(has_right_mask)) +
                         [1] * int(cp.sum(has_right_mask))).astype(cp.float32)
    row_idx = pixel_idx[has_right_mask]
    row_idx = cp.tile(row_idx, 2)
    col_idx = cp.concatenate(
        (pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]))
    D_horizontal_pos = coo_matrix((data_term, (row_idx, col_idx)),
                                  shape=(num_pixel, num_pixel))

    data_term = cp.array([-1] * int(cp.sum(has_top_mask)) +
                         [1] * int(cp.sum(has_top_mask))).astype(cp.float32)
    row_idx = pixel_idx[has_top_mask]
    row_idx = cp.tile(row_idx, 2)
    col_idx = cp.concatenate(
        (pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]))
    D_vertical_pos = coo_matrix((data_term, (row_idx, col_idx)),
                                shape=(num_pixel, num_pixel))

    data_term = cp.array([-1] * int(cp.sum(has_bottom_mask)) +
                         [1] * int(cp.sum(has_bottom_mask))).astype(cp.float32)
    row_idx = pixel_idx[has_bottom_mask]
    row_idx = cp.tile(row_idx, 2)
    col_idx = cp.concatenate(
        (pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]))
    D_vertical_neg = coo_matrix((data_term, (row_idx, col_idx)),
                                shape=(num_pixel, num_pixel))

    return (
        D_horizontal_pos / step_size,
        D_horizontal_neg / step_size,
        D_vertical_pos / step_size,
        D_vertical_neg / step_size,
    )


def construct_facets_from(mask):
    idx = cp.zeros_like(mask, dtype=int)
    idx[mask] = cp.arange(cp.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)
    facet_top_left_mask = (facet_move_top_mask * facet_move_left_mask *
                           facet_move_top_left_mask * mask)
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
        u = u[mask].T  # 3 x m
        vertices = (cp.linalg.inv(K) @ u).T * depth_map[mask,
                                                        cp.newaxis]  # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def bilateral_normal_integration(
    normal_map,
    normal_mask,
    k=2,
    lambda1=0,
    depth_map=None,
    depth_mask=None,
    K=None,
    step_size=1,
    max_iter=100,
    tol=1e-5,
    cg_max_iter=500,
    cg_tol=1e-5,
    label="",
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

    # print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
    #       f"The number of normal vectors is {cp.sum(normal_mask)}.")

    # transfer the normal map from the normal coordinates to the camera coordinates

    normal_map = cp.asarray(normal_map)
    normal_mask = cp.asarray(normal_mask)
    depth_map = cp.asarray(depth_map)
    depth_mask = cp.asarray(depth_mask)

    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = -normal_map[normal_mask, 2]

    if K is not None:  # perspective
        H, W = normal_mask.shape

        yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
        xx = cp.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        Nz_u = diags(uu * nx + vv * ny + fx * nz)
        Nz_v = diags(uu * nx + vv * ny + fy * nz)

    else:  # orthographic
        Nz_u = diags(nz)
        Nz_v = diags(nz)

    # get partial derivative matrices
    Dvp, Dvn, Dup, Dun = generate_dx_dy(normal_mask, step_size)

    A1 = Nz_u @ Dup
    A2 = Nz_u @ Dun
    A3 = Nz_v @ Dvp
    A4 = Nz_v @ Dvn

    A = vstack((A1, A2, A3, A4))
    b = cp.concatenate((-nx, -nx, -ny, -ny))

    # initialization
    W = 0.5 * diags(cp.ones_like(b))
    z = cp.zeros(cp.sum(normal_mask).item())
    energy = (A @ z - b).T @ W @ (A @ z - b)

    energy_list = []

    m = depth_mask[normal_mask].astype(int)  # shape: (num_normals,)
    M = diags(m)

    z_prior = (cp.log(depth_map)[normal_mask] if K is not None else
               depth_map[normal_mask])  # shape: (num_normals,)

    pbar = tqdm(range(max_iter))

    for i in pbar:

        depth_diff = M @ (z_prior - z)
        depth_diff[depth_diff == 0] = cp.nan
        offset = cp.nanmean(depth_diff)
        z = z + offset

        A_mat = A.T @ W @ A + lambda1 * M
        b_mat = A.T @ W @ b + lambda1 * M @ z_prior
        z, _ = cg(A_mat, b_mat, x0=z, maxiter=cg_max_iter, tol=cg_tol)

        # update weights
        wu = sigmoid((A2 @ z)**2 - (A1 @ z)**2, k)
        wv = sigmoid((A4 @ z)**2 - (A3 @ z)**2, k)
        W = diags(cp.concatenate((wu, 1 - wu, wv, 1 - wv)))

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"BNI[{label}] steps --- {i+1}/{max_iter} energy: {energy} relative energy: {relative_energy:.3e}"
        )
        if relative_energy < tol:
            break

    depth_map = cp.ones_like(normal_mask, float) * cp.nan
    depth_map[normal_mask] = z

    if K is not None:  # perspective
        depth_map = cp.exp(depth_map)
        vertices = cp.asnumpy(
            map_depth_map_to_point_clouds(depth_map, normal_mask, K=K))
    else:  # orthographic
        vertices = cp.asnumpy(
            map_depth_map_to_point_clouds(depth_map,
                                          normal_mask,
                                          K=None,
                                          step_size=step_size))

    faces = cp.asnumpy(construct_facets_from(normal_mask))

    if normal_map[:, :, -1].mean() < 0:
        faces = np.concatenate((faces[:, [1, 4, 3]], faces[:, [1, 3, 2]]),
                               axis=0)
    else:
        faces = np.concatenate((faces[:, [1, 2, 3]], faces[:, [1, 3, 4]]),
                               axis=0)

    vertices, faces = remove_stretched_faces(vertices, faces)

    return torch.as_tensor(vertices), torch.as_tensor(faces).long()
