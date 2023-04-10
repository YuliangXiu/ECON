# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import torch.nn as nn
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from tqdm import tqdm

from lib.common.train_util import init_loss
from lib.dataset.mesh_util import update_mesh_shape_prior_losses


# reference: https://github.com/wuhaozhe/pytorch-nicp
class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size=1, edges=None):
        '''
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        '''
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(
            torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)
        )
        self.b = nn.Parameter(
            torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(
                batch_size, num_points, 1, 1
            )
        )
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        '''
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix, 
        '''
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=self.edges[:, 0])
        w2 = torch.index_select(affine_weight, dim=1, index=self.edges[:, 1])
        w_diff = (w1 - w2)**2
        w_rigid = (torch.linalg.det(self.A) - 1.0)**2
        return w_diff, w_rigid

    def forward(self, x):
        '''
            x should have shape of B * N * 3 * 1
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        stiffness, rigid = self.stiffness()

        return out_x, stiffness, rigid


def trimesh2meshes(mesh):
    '''
        convert trimesh mesh to pytorch3d mesh
    '''
    verts = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    mesh = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
    return mesh


def register(target_mesh, src_mesh, device, verbose=True):

    # define local_affine deform verts
    tgt_mesh = trimesh2meshes(target_mesh).to(device)
    src_verts = src_mesh.verts_padded().clone()

    local_affine_model = LocalAffine(
        src_mesh.verts_padded().shape[1],
        src_mesh.verts_padded().shape[0], src_mesh.edges_packed()
    ).to(device)

    optimizer_cloth = torch.optim.Adam([{'params': local_affine_model.parameters()}],
                                       lr=1e-2,
                                       amsgrad=True)
    scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cloth,
        mode="min",
        factor=0.1,
        verbose=0,
        min_lr=1e-5,
        patience=5,
    )

    losses = init_loss()

    if verbose:
        loop_cloth = tqdm(range(100))
    else:
        loop_cloth = range(100)

    for i in loop_cloth:

        optimizer_cloth.zero_grad()

        deformed_verts, stiffness, rigid = local_affine_model(x=src_verts)
        src_mesh = src_mesh.update_padded(deformed_verts)

        # losses for laplacian, edge, normal consistency
        update_mesh_shape_prior_losses(src_mesh, losses)

        losses["cloth"]["value"] = chamfer_distance(
            x=src_mesh.verts_padded(), y=tgt_mesh.verts_padded()
        )[0]
        losses["stiff"]["value"] = torch.mean(stiffness)
        losses["rigid"]["value"] = torch.mean(rigid)

        # Weighted sum of the losses
        cloth_loss = torch.tensor(0.0, requires_grad=True).to(device)
        pbar_desc = "Register SMPL-X -> d-BiNI -- "

        for k in losses.keys():
            if losses[k]["weight"] > 0.0 and losses[k]["value"] != 0.0:
                cloth_loss = cloth_loss + \
                    losses[k]["value"] * losses[k]["weight"]
                pbar_desc += f"{k}:{losses[k]['value']* losses[k]['weight']:.3f} | "

        if verbose:
            pbar_desc += f"TOTAL: {cloth_loss:.3f}"
            loop_cloth.set_description(pbar_desc)

        # update params
        cloth_loss.backward(retain_graph=True)
        optimizer_cloth.step()
        scheduler_cloth.step(cloth_loss)

    final = trimesh.Trimesh(
        src_mesh.verts_packed().detach().squeeze(0).cpu(),
        src_mesh.faces_packed().detach().squeeze(0).cpu(),
        process=False,
        maintains_order=True
    )

    return final
