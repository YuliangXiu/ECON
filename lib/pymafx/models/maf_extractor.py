# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py

import logging

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pymafx.core import path_config
from lib.pymafx.utils.geometry import projection

logger = logging.getLogger(__name__)

from lib.pymafx.utils.imutils import j2d_processing

from .transformers.net_utils import PosEnSine
from .transformers.transformer_basics import OurMultiheadAttention


class TransformerDecoderUnit(nn.Module):
    def __init__(
        self, feat_dim, attri_dim=0, n_head=8, pos_en_flag=True, attn_type='softmax', P=None
    ):
        super(TransformerDecoderUnit, self).__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        assert attri_dim == 0
        if self.pos_en_flag:
            pe_dim = 10
            self.pos_en = PosEnSine(pe_dim)
        else:
            pe_dim = 0
        self.attn = OurMultiheadAttention(
            feat_dim + attri_dim + pe_dim * 3, feat_dim + pe_dim * 3, feat_dim, n_head
        )    # cross-attention

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(self.feat_dim)

    def forward(self, q, k, v, pos=None):
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q, pos)
            k_pos_embed = self.pos_en(k)

            q = torch.cat([q, q_pos_embed], dim=1)
            k = torch.cat([k, k_pos_embed], dim=1)
        # else:
        #     q_pos_embed = 0
        #     k_pos_embed = 0

        # cross-multi-head attention
        out = self.attn(q=q, k=k, v=v, attn_type=self.attn_type, P=self.P)[0]

        # feed forward
        out2 = self.linear2(self.activation(self.linear1(out)))
        out = out + out2
        out = self.norm(out)

        return out


class Mesh_Sampler(nn.Module):
    ''' Mesh Up/Down-sampling
    '''
    def __init__(self, type='smpl', level=2, device=torch.device('cuda'), option=None):
        super().__init__()

        # downsample SMPL mesh and assign part labels
        if type == 'smpl':
            # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
            smpl_mesh_graph = np.load(
                path_config.SMPL_DOWNSAMPLING, allow_pickle=True, encoding='latin1'
            )

            A = smpl_mesh_graph['A']
            U = smpl_mesh_graph['U']
            D = smpl_mesh_graph['D']    # shape: (2,)
        elif type == 'mano':
            # from https://github.com/microsoft/MeshGraphormer/blob/main/src/modeling/data/mano_downsampling.npz
            mano_mesh_graph = np.load(
                path_config.MANO_DOWNSAMPLING, allow_pickle=True, encoding='latin1'
            )

            A = mano_mesh_graph['A']
            U = mano_mesh_graph['U']
            D = mano_mesh_graph['D']    # shape: (2,)

        # downsampling
        ptD = []
        for lv in range(len(D)):
            d = scipy.sparse.coo_matrix(D[lv])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890] , [195, 778]
        # ptD[1].to_dense() - Size: [431, 1723] , [49, 195]
        if level == 2:
            Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense())    # 6890 -> 431
        elif level == 1:
            Dmap = ptD[0].to_dense()    #
        self.register_buffer('Dmap', Dmap)

        # upsampling
        ptU = []
        for lv in range(len(U)):
            d = scipy.sparse.coo_matrix(U[lv])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptU.append(torch.sparse.FloatTensor(i, v, d.shape))

        # upsampling mapping from 431 points to 6890 points
        # ptU[0].to_dense() - Size: [6890, 1723]
        # ptU[1].to_dense() - Size: [1723, 431]
        if level == 2:
            Umap = torch.matmul(ptU[0].to_dense(), ptU[1].to_dense())    # 431 -> 6890
        elif level == 1:
            Umap = ptU[0].to_dense()    #
        self.register_buffer('Umap', Umap)

    def downsample(self, x):
        return torch.matmul(self.Dmap.unsqueeze(0), x)    # [B, 431, 3]

    def upsample(self, x):
        return torch.matmul(self.Umap.unsqueeze(0), x)    # [B, 6890, 3]

    def forward(self, x, mode='downsample'):
        if mode == 'downsample':
            return self.downsample(x)
        elif mode == 'upsample':
            return self.upsample(x)


class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator
    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''
    def __init__(
        self, filter_channels, device=torch.device('cuda'), iwp_cam_mode=True, option=None
    ):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        self.last_op = nn.ReLU(True)

        self.iwp_cam_mode = iwp_cam_mode

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1)
                )
            else:
                self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            self.add_module("conv%d" % l, self.filters[l])

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load(
            path_config.SMPL_DOWNSAMPLING, allow_pickle=True, encoding='latin1'
        )

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D']    # shape: (2,)

        # downsampling
        ptD = []
        for level in range(len(D)):
            d = scipy.sparse.coo_matrix(D[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense())    # 6890 -> 431
        self.register_buffer('Dmap', Dmap)

        # upsampling
        ptU = []
        for level in range(len(U)):
            d = scipy.sparse.coo_matrix(U[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptU.append(torch.sparse.FloatTensor(i, v, d.shape))

        # upsampling mapping from 431 points to 6890 points
        # ptU[0].to_dense() - Size: [6890, 1723]
        # ptU[1].to_dense() - Size: [1723, 431]
        Umap = torch.matmul(ptU[0].to_dense(), ptU[1].to_dense())    # 431 -> 6890
        self.register_buffer('Umap', Umap)

    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2]).mean(dim=1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1],
                                    feature.shape[2]).mean(dim=1)

        y = self.last_op(y)

        # y = y.view(y.shape[0], -1)

        return y

    def sampling(self, points, im_feat=None, z_feat=None, add_att=False, reduce_dim=True):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        # if im_feat is None:
        #     im_feat = self.im_feat

        batch_size = im_feat.shape[0]
        point_feat = torch.nn.functional.grid_sample(
            im_feat, points.unsqueeze(2), align_corners=False
        )[..., 0]

        if reduce_dim:
            mesh_align_feat = self.reduce_dim(point_feat)
            return mesh_align_feat
        else:
            return point_feat

    def forward(self, p, im_feat, cam=None, add_att=False, reduce_dim=True, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.
        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            im_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        # if cam is None:
        #     cam = self.cam
        p_proj_2d = projection(p, cam, retain_z=False, iwp_mode=self.iwp_cam_mode)
        if self.iwp_cam_mode:
            # Normalize keypoints to [-1,1]
            p_proj_2d = p_proj_2d / (224. / 2.)
        else:
            p_proj_2d = j2d_processing(p_proj_2d, cam['kps_transf'])
        mesh_align_feat = self.sampling(p_proj_2d, im_feat, add_att=add_att, reduce_dim=reduce_dim)
        return mesh_align_feat
