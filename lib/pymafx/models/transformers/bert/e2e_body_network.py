"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg


class Graphormer_Body_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super(Graphormer_Body_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(431, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1, 72))
        template_pose[:, 0] = 3.1416    # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1, 10)).cuda(self.config.device)
        template_vertices = smpl(template_pose, template_betas)

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)
        print('template_vertices', template_vertices.shape, template_vertices_sub.shape, template_vertices_sub2.shape)

        # template mesh-to-joint regression
        template_3d_joints = smpl.get_h36m_joints(template_vertices)
        template_pelvis = template_3d_joints[:, cfg.H36M_J17_NAME.index('Pelvis'), :]
        template_3d_joints = template_3d_joints[:, cfg.H36M_J17_TO_J14, :]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2], dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)
        print('ref_vertices', ref_vertices.shape)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        print('image_feat, grid_feat', image_feat.shape, grid_feat.shape)
        # concatinate image feat and 3d mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        print('image_feat', image_feat.shape)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)
        print('grid_feat bf', grid_feat.shape)
        grid_feat = self.grid_feat_dim(grid_feat)
        print('grid_feat', grid_feat.shape)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, image_feat], dim=2)
        print('features', features.shape, ref_vertices.shape, image_feat.shape)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat], dim=1)
        print('features', features.shape)

        if is_train == True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            special_token = torch.ones_like(features[:, :-49, :]).cuda() * 0.01
            print('special_token', special_token.shape, meta_masks.shape)
            print('meta_masks', torch.unique(meta_masks))
            features[:, :-49, :] = features[:, :-49, :] * meta_masks + special_token * (1 - meta_masks)

        # forward pass
        if self.config.output_attentions == True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:, :num_joints, :]
        pred_vertices_sub2 = features[:, num_joints:-49, :]

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1, 2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1, 2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1, 2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)
        pred_vertices_full = pred_vertices_full.transpose(1, 2)

        if self.config.output_attentions == True:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, hidden_states, att
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full
