import logging
from copy import deepcopy

import numpy as np
import torch

from .utils.libkdtree import KDTree

logger_py = logging.getLogger(__name__)


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def rgb2gray(rgb):
    ''' rgb of size B x h x w x 3
    '''
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def sample_patch_points(
    batch_size, n_points, patch_size=1, image_resolution=(128, 128), continuous=True
):
    ''' Returns sampled points in the range [-1, 1].

    Args:
        batch_size (int): required batch size
        n_points (int): number of points to sample
        patch_size (int): size of patch; if > 1, patches of size patch_size
            are sampled instead of individual points
        image_resolution (tuple): image resolution (required for calculating
            the pixel distances)
        continuous (bool): whether to sample continuously or only on pixel
            locations
    '''
    assert (patch_size > 0)
    # Calculate step size for [-1, 1] that is equivalent to a pixel in
    # original resolution
    h_step = 1. / image_resolution[0]
    w_step = 1. / image_resolution[1]
    # Get number of patches
    patch_size_squared = patch_size**2
    n_patches = int(n_points / patch_size_squared)
    if continuous:
        p = torch.rand(batch_size, n_patches, 2)    # [0, 1]
    else:
        px = torch.randint(0, image_resolution[1],
                           size=(batch_size, n_patches, 1)).float() / (image_resolution[1] - 1)
        py = torch.randint(0, image_resolution[0],
                           size=(batch_size, n_patches, 1)).float() / (image_resolution[0] - 1)
        p = torch.cat([px, py], dim=-1)
    # Scale p to [0, (1 - (patch_size - 1) * step) ]
    p[:, :, 0] *= 1 - (patch_size - 1) * w_step
    p[:, :, 1] *= 1 - (patch_size - 1) * h_step

    # Add points
    patch_arange = torch.arange(patch_size)
    x_offset, y_offset = torch.meshgrid(patch_arange, patch_arange)
    patch_offsets = torch.stack([x_offset.reshape(-1), y_offset.reshape(-1)],
                                dim=1).view(1, 1, -1, 2).repeat(batch_size, n_patches, 1, 1).float()

    patch_offsets[:, :, :, 0] *= w_step
    patch_offsets[:, :, :, 1] *= h_step

    # Add patch_offsets to points
    p = p.view(batch_size, n_patches, 1, 2) + patch_offsets

    # Scale to [-1, x]
    p = p * 2 - 1

    p = p.view(batch_size, -1, 2)

    amax, amin = p.max(), p.min()
    assert (amax <= 1. and amin >= -1.)

    return p


def get_proposal_points_in_unit_cube(ray0, ray_direction, padding=0.1, eps=1e-6, n_steps=40):
    ''' Returns n_steps equally spaced points inside the unit cube on the rays
    cast from ray0 with direction ray_direction.

    This function is used to get the ray marching points {p^ray_j} for a given
    camera position ray0 and
    a given ray direction ray_direction which goes from the camera_position to
    the pixel location.

    NOTE: The returned values d_proposal are the lengths of the ray:
        p^ray_j = ray0 + d_proposal_j * ray_direction

    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
        n_steps (int): number of steps
    '''
    batch_size, n_pts, _ = ray0.shape
    device = ray0.device

    p_intervals, d_intervals, mask_inside_cube = \
        check_ray_intersection_with_unit_cube(ray0, ray_direction, padding,
                                              eps)
    d_proposal = d_intervals[:, :, 0].unsqueeze(-1) + \
        torch.linspace(0, 1, steps=n_steps).to(device).view(1, 1, -1) * \
        (d_intervals[:, :, 1] - d_intervals[:, :, 0]).unsqueeze(-1)
    d_proposal = d_proposal.unsqueeze(-1)

    return d_proposal, mask_inside_cube


def check_ray_intersection_with_unit_cube(ray0, ray_direction, padding=0.1, eps=1e-6, scale=2.0):
    ''' Checks if rays ray0 + d * ray_direction intersect with unit cube with
    padding padding.

    It returns the two intersection points as well as the sorted ray lengths d.

    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
        scale (float): cube size
    '''
    batch_size, n_pts, _ = ray0.shape
    device = ray0.device

    # calculate intersections with unit cube (< . , . >  is the dot product)
    # <n, x - p> = <n, ray0 + d * ray_direction - p_e> = 0
    # d = - <n, ray0 - p_e> / <n, ray_direction>

    # Get points on plane p_e
    p_distance = (scale * 0.5) + padding / 2
    p_e = torch.ones(batch_size, n_pts, 6).to(device) * p_distance
    p_e[:, :, 3:] *= -1.

    # Calculate the intersection points with given formula
    nominator = p_e - ray0.repeat(1, 1, 2)
    denominator = ray_direction.repeat(1, 1, 2)
    d_intersect = nominator / denominator
    p_intersect = ray0.unsqueeze(-2) + d_intersect.unsqueeze(-1) * \
        ray_direction.unsqueeze(-2)

    # Calculate mask where points intersect unit cube
    p_mask_inside_cube = ((p_intersect[:, :, :, 0] <= p_distance + eps) &
                          (p_intersect[:, :, :, 1] <= p_distance + eps) &
                          (p_intersect[:, :, :, 2] <= p_distance + eps) &
                          (p_intersect[:, :, :, 0] >= -(p_distance + eps)) &
                          (p_intersect[:, :, :, 1] >= -(p_distance + eps)) &
                          (p_intersect[:, :, :, 2] >= -(p_distance + eps))).cpu()

    # Correct rays are these which intersect exactly 2 times
    mask_inside_cube = p_mask_inside_cube.sum(-1) == 2

    # Get interval values for p's which are valid
    p_intervals = p_intersect[mask_inside_cube][p_mask_inside_cube[mask_inside_cube]].view(-1, 2, 3)
    p_intervals_batch = torch.zeros(batch_size, n_pts, 2, 3).to(device)
    p_intervals_batch[mask_inside_cube] = p_intervals

    # Calculate ray lengths for the interval points
    d_intervals_batch = torch.zeros(batch_size, n_pts, 2).to(device)
    norm_ray = torch.norm(ray_direction[mask_inside_cube], dim=-1)
    d_intervals_batch[mask_inside_cube] = torch.stack([
        torch.norm(p_intervals[:, 0] - ray0[mask_inside_cube], dim=-1) / norm_ray,
        torch.norm(p_intervals[:, 1] - ray0[mask_inside_cube], dim=-1) / norm_ray,
    ],
                                                      dim=-1)

    # Sort the ray lengths
    d_intervals_batch, indices_sort = d_intervals_batch.sort()
    p_intervals_batch = p_intervals_batch[torch.arange(batch_size).view(-1, 1, 1),
                                          torch.arange(n_pts).view(1, -1, 1), indices_sort]

    return p_intervals_batch, d_intervals_batch, mask_inside_cube


def intersect_camera_rays_with_unit_cube(
    pixels, camera_mat, world_mat, scale_mat, padding=0.1, eps=1e-6, use_ray_length_as_depth=True
):
    ''' Returns the intersection points of ray cast from camera origin to
    pixel points p on the image plane.

    The function returns the intersection points as well the depth values and
    a mask specifying which ray intersects the unit cube.

    Args:
        pixels (tensor): Pixel points on image plane (range [-1, 1])
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability

    '''
    batch_size, n_points, _ = pixels.shape

    pixel_world = image_points_to_world(pixels, camera_mat, world_mat, scale_mat)
    camera_world = origin_to_world(n_points, camera_mat, world_mat, scale_mat)
    ray_vector = (pixel_world - camera_world)

    p_cube, d_cube, mask_cube = check_ray_intersection_with_unit_cube(
        camera_world, ray_vector, padding=padding, eps=eps
    )
    if not use_ray_length_as_depth:
        p_cam = transform_to_camera_space(
            p_cube.view(batch_size, -1, 3), camera_mat, world_mat, scale_mat
        ).view(batch_size, n_points, -1, 3)
        d_cube = p_cam[:, :, :, -1]
    return p_cube, d_cube, mask_cube


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.), subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack([pixel_locations[0], pixel_locations[1]],
                                  dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to, ), replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.to(torch.bool)
    if is_numpy:
        mask = mask.numpy()

    return mask


def transform_mesh(mesh, transform):
    ''' Transforms a mesh with given transformation.

    Args:
        mesh (trimesh mesh): mesh
        transform (tensor): transformation matrix of size 4 x 4
    '''
    mesh = deepcopy(mesh)
    v = np.asarray(mesh.vertices).astype(np.float32)
    v_transformed = transform_pointcloud(v, transform)
    mesh.vertices = v_transformed
    return mesh


def transform_pointcloud(pointcloud, transform):
    ''' Transforms a point cloud with given transformation.

    Args:
        pointcloud (tensor): tensor of size N x 3
        transform (tensor): transformation of size 4 x 4
    '''

    assert (transform.shape == (4, 4) and pointcloud.shape[-1] == 3)

    pcl, is_numpy = to_pytorch(pointcloud, True)
    transform = to_pytorch(transform)

    # Transform point cloud to homogen coordinate system
    pcl_hom = torch.cat([pcl, torch.ones(pcl.shape[0], 1)], dim=-1).transpose(1, 0)

    # Apply transformation to point cloud
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:3].transpose(1, 0)
    if is_numpy:
        pcl_out = pcl_out.numpy()

    return pcl_out


def transform_points_batch(p, transform):
    ''' Transform points tensor with given transform.

    Args:
        p (tensor): tensor of size B x N x 3
        transform (tensor): transformation of size B x 4 x 4
    '''
    device = p.device
    assert (transform.shape[1:] == (4, 4) and p.shape[-1] == 3 and p.shape[0] == transform.shape[0])

    # Transform points to homogen coordinates
    pcl_hom = torch.cat([p, torch.ones(p.shape[0], p.shape[1], 1).to(device)],
                        dim=-1).transpose(2, 1)

    # Apply transformation
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:, :3].transpose(2, 1)
    return pcl_out


def get_tensor_values(
    tensor, p, grid_sample=True, mode='nearest', with_mask=False, squeeze_channel_dim=False
):
    '''
    Returns values from tensor at given location p.

    Args:
        tensor (tensor): tensor of size B x C x H x W
        p (tensor): position values scaled between [-1, 1] and
            of size B x N x 2
        grid_sample (boolean): whether to use grid sampling
        mode (string): what mode to perform grid sampling in
        with_mask (bool): whether to return the mask for invalid values
        squeeze_channel_dim (bool): whether to squeeze the channel dimension
            (only applicable to 1D data)
    '''
    p = to_pytorch(p)
    tensor, is_numpy = to_pytorch(tensor, True)
    batch_size, _, h, w = tensor.shape

    if grid_sample:
        p = p.unsqueeze(1)
        values = torch.nn.functional.grid_sample(tensor, p, mode=mode)
        values = values.squeeze(2)
        values = values.permute(0, 2, 1)
    else:
        p[:, :, 0] = (p[:, :, 0] + 1) * (w) / 2
        p[:, :, 1] = (p[:, :, 1] + 1) * (h) / 2
        p = p.long()
        values = tensor[torch.arange(batch_size).unsqueeze(-1), :, p[:, :, 1], p[:, :, 0]]

    if with_mask:
        mask = get_mask(values)
        if squeeze_channel_dim:
            mask = mask.squeeze(-1)
        if is_numpy:
            mask = mask.numpy()

    if squeeze_channel_dim:
        values = values.squeeze(-1)

    if is_numpy:
        values = values.numpy()

    if with_mask:
        return values, mask
    return values


def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert (pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat([p_world, torch.ones(batch_size, n_p, 1).to(device)],
                        dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device

    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert (dim == 2)
    device = image_points.device

    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    return transform_to_world(
        image_points, d_image, camera_mat, world_mat, scale_mat, invert=invert
    )


def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)


def check_tensor(tensor, tensorname='', input_tensor=None):
    ''' Checks tensor for illegal values.

    Args:
        tensor (tensor): tensor
        tensorname (string): name of tensor
        input_tensor (tensor): previous input
    '''
    if torch.isnan(tensor).any():
        logger_py.warn('Tensor %s contains nan values.' % tensorname)
        if input_tensor is not None:
            logger_py.warn(f'Input was: {input_tensor}')


def get_prob_from_logits(logits):
    ''' Returns probabilities for logits

    Args:
        logits (tensor): logits
    '''
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs


def get_logits_from_prob(probs, eps=1e-4):
    ''' Returns logits for probabilities.

    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = np.clip(probs, a_min=eps, a_max=1 - eps)
    logits = np.log(probs / (1 - probs))
    return logits


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''
    if use_kdtree:
        return chamfer_distance_kdtree(points1, points2, give_id=give_id)
    else:
        return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
    '''
    assert (points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indices
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indices
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def get_occupancy_loss_points(
    pixels,
    camera_mat,
    world_mat,
    scale_mat,
    depth_image=None,
    use_cube_intersection=True,
    occupancy_random_normal=False,
    depth_range=[0, 2.4]
):
    ''' Returns 3D points for occupancy loss.

    Args:
        pixels (tensor): sampled pixels in range [-1, 1]
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        depth_image tensor): if not None, these depth values are used for
            initialization (e.g. depth or visual hull depth)
        use_cube_intersection (bool): whether to check unit cube intersection
        occupancy_random_normal (bool): whether to sample from a Normal
            distribution instead of a uniform one
        depth_range (float): depth range; important when no cube
            intersection is used
    '''
    device = pixels.device
    batch_size, n_points, _ = pixels.shape

    if use_cube_intersection:
        _, d_cube_intersection, mask_cube = \
            intersect_camera_rays_with_unit_cube(
                pixels, camera_mat, world_mat, scale_mat, padding=0.,
                use_ray_length_as_depth=False)
        d_cube = d_cube_intersection[mask_cube]

    d_occupancy = torch.rand(batch_size, n_points).to(device) * depth_range[1]

    if use_cube_intersection:
        d_occupancy[mask_cube] = d_cube[:, 0] + \
            torch.rand(d_cube.shape[0]).to(
                device) * (d_cube[:, 1] - d_cube[:, 0])
    if occupancy_random_normal:
        d_occupancy = torch.randn(batch_size, n_points).to(device) \
            * (depth_range[1] / 8) + depth_range[1] / 2
        if use_cube_intersection:
            mean_cube = d_cube.sum(-1) / 2
            std_cube = (d_cube[:, 1] - d_cube[:, 0]) / 8
            d_occupancy[mask_cube] = mean_cube + \
                torch.randn(mean_cube.shape[0]).to(device) * std_cube

    if depth_image is not None:
        depth_gt, mask_gt_depth = get_tensor_values(
            depth_image, pixels, squeeze_channel_dim=True, with_mask=True
        )
        d_occupancy[mask_gt_depth] = depth_gt[mask_gt_depth]

    p_occupancy = transform_to_world(
        pixels, d_occupancy.unsqueeze(-1), camera_mat, world_mat, scale_mat
    )
    return p_occupancy


def get_freespace_loss_points(
    pixels, camera_mat, world_mat, scale_mat, use_cube_intersection=True, depth_range=[0, 2.4]
):
    ''' Returns 3D points for freespace loss.

    Args:
        pixels (tensor): sampled pixels in range [-1, 1]
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        use_cube_intersection (bool): whether to check unit cube intersection
        depth_range (float): depth range; important when no cube
            intersection is used
    '''
    device = pixels.device
    batch_size, n_points, _ = pixels.shape

    d_freespace = torch.rand(batch_size, n_points).to(device) * \
        depth_range[1]

    if use_cube_intersection:
        _, d_cube_intersection, mask_cube = \
            intersect_camera_rays_with_unit_cube(
                pixels, camera_mat, world_mat, scale_mat,
                use_ray_length_as_depth=False)
        d_cube = d_cube_intersection[mask_cube]
        d_freespace[mask_cube] = d_cube[:, 0] + \
            torch.rand(d_cube.shape[0]).to(
                device) * (d_cube[:, 1] - d_cube[:, 0])

    p_freespace = transform_to_world(
        pixels, d_freespace.unsqueeze(-1), camera_mat, world_mat, scale_mat
    )
    return p_freespace


def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
    ''' Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    '''
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True), min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor
