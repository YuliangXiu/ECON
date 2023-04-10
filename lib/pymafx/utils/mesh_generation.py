import time

import numpy as np
import torch
import torch.optim as optim
import trimesh
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from .common import make_3d_grid, transform_pointcloud
from .utils import libmcubes
from .utils.libmise import MISE
from .utils.libsimplify import simplify_mesh


class Generator3D(object):
    '''  Generator class for DVRs.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained DVR model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        simplify_nfaces (int): number of faces the mesh should be simplified to
        refine_max_faces (int): max number of faces which are used as batch
            size for refinement process (we added this functionality in this
            work)
    '''
    def __init__(
        self,
        model,
        points_batch_size=100000,
        threshold=0.5,
        refinement_step=0,
        device=None,
        resolution0=16,
        upsampling_steps=3,
        with_normals=False,
        padding=0.1,
        simplify_nfaces=None,
        with_color=False,
        refine_max_faces=10000
    ):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.simplify_nfaces = simplify_nfaces
        self.with_color = with_color
        self.refine_max_faces = refine_max_faces

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        c = self.model.encode_inputs(inputs)
        mesh = self.generate_from_latent(c, stats_dict=stats_dict, data=data, **kwargs)

        return mesh, stats_dict

    def generate_meshes(self, data, return_stats=True):
        ''' Generates the output meshes with data of batch size >=1

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        meshes = []
        for i in range(inputs.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            c = self.model.encode_inputs(input_i)
            mesh = self.generate_from_latent(c, stats_dict=stats_dict)
            meshes.append(mesh)

        return meshes

    def generate_pointcloud(self, mesh, data=None, n_points=2000000, scale_back=True):
        ''' Generates a point cloud from the mesh.

        Args:
            mesh (trimesh): mesh
            data (dict): data dictionary
            n_points (int): number of point cloud points
            scale_back (bool): whether to undo scaling (requires a scale
                matrix in data dictionary)
        '''
        pcl = mesh.sample(n_points).astype(np.float32)

        if scale_back:
            scale_mat = data.get('camera.scale_mat_0', None)
            if scale_mat is not None:
                pcl = transform_pointcloud(pcl, scale_mat[0])
            else:
                print('Warning: No scale_mat found!')
        pcl_out = trimesh.Trimesh(vertices=pcl, process=False)
        return pcl_out

    def generate_from_latent(self, c=None, pl=None, stats_dict={}, data=None, **kwargs):
        ''' Generates mesh from latent.

        Args:
            c (tensor): latent conditioned code c
            pl (tensor): predicted plane parameters
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid((-0.5, ) * 3, (0.5, ) * 3, (nx, ) * 3)
            values = self.eval_points(pointsf, c, pl, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = 2 * pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 1.0)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, pl, **kwargs).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, pl=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, c, pl, **kwargs).logits

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices *= 2
        vertices = box_size * (vertices - 1)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0
        else:
            normals = None
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices,
            triangles,
            vertex_normals=normals,
        # vertex_colors=vertex_colors,
            process=False
        )

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        # Estimate Vertex Colors
        if self.with_color and not vertices.shape[0] == 0:
            t0 = time.time()
            vertex_colors = self.estimate_colors(np.array(mesh.vertices), c)
            stats_dict['time (color)'] = time.time() - t0
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors,
                process=False
            )

        return mesh

    def estimate_colors(self, vertices, c=None):
        ''' Estimates vertex colors by evaluating the texture field.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)
        colors = []
        for vi in vertices_split:
            vi = vi.to(device)
            with torch.no_grad():
                ci = self.model.decode_color(vi.unsqueeze(0), c).squeeze(0).cpu()
            colors.append(ci)

        colors = np.concatenate(colors, axis=0)
        colors = np.clip(colors, 0, 1)
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)],
                                axis=1)
        return colors

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert (n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces)

        # detach c; otherwise graph needs to be retained
        # caused by new Pytorch version?
        c = c.detach()

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-5)

        # Dataset
        ds_faces = TensorDataset(faces)
        dataloader = DataLoader(ds_faces, batch_size=self.refine_max_faces, shuffle=True)

        # We updated the refinement algorithm to subsample faces; this is
        # usefull when using a high extraction resolution / when working on
        # small GPUs
        it_r = 0
        while it_r < self.refinement_step:
            for f_it in dataloader:
                f_it = f_it[0].to(self.device)
                optimizer.zero_grad()

                # Loss
                face_vertex = v[f_it]
                eps = np.random.dirichlet((0.5, 0.5, 0.5), size=f_it.shape[0])
                eps = torch.FloatTensor(eps).to(self.device)
                face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

                face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
                face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
                face_normal = torch.cross(face_v1, face_v2)
                face_normal = face_normal / \
                    (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                face_value = torch.cat([
                    torch.sigmoid(self.model.decode(p_split, c).logits)
                    for p_split in torch.split(face_point.unsqueeze(0), 20000, dim=1)
                ],
                                       dim=1)

                normal_target = -autograd.grad([face_value.sum()], [face_point],
                                               create_graph=True)[0]

                normal_target = \
                    normal_target / \
                    (normal_target.norm(dim=1, keepdim=True) + 1e-10)
                loss_target = (face_value - threshold).pow(2).mean()
                loss_normal = \
                    (face_normal - normal_target).pow(2).sum(dim=1).mean()

                loss = loss_target + 0.01 * loss_normal

                # Update
                loss.backward()
                optimizer.step()

                # Update it_r
                it_r += 1

                if it_r >= self.refinement_step:
                    break

        mesh.vertices = v.data.cpu().numpy()
        return mesh
