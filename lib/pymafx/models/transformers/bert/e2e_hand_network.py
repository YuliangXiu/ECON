"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import src.modeling.data.config as cfg


class Graphormer_Hand_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for hand pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder):
        super(Graphormer_Hand_Network, self).__init__()
        self.config = config
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(195, 778)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(195 + 21, 150)
        self.cam_param_fc3 = torch.nn.Linear(150, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

    def forward(self, images, mesh_model, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1, 48))
        template_pose = template_pose.cuda()
        template_betas = torch.zeros((1, 10)).cuda()
        template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
        template_vertices = template_vertices / 1000.0
        template_3d_joints = template_3d_joints / 1000.0

        template_vertices_sub = mesh_sampler.downsample(template_vertices)

        # normalize
        template_root = template_3d_joints[:, cfg.J_NAME.index('Wrist'), :]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        template_vertices_sub = template_vertices_sub - template_root[:, None, :]
        num_joints = template_3d_joints.shape[1]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # concatinate image feat and mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)
        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, image_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat], dim=1)

        if is_train == True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            special_token = torch.ones_like(features[:, :-49, :]).cuda() * 0.01
            features[:, :-49, :] = features[:, :-49, :] * meta_masks + special_token * (1 - meta_masks)

        # forward pass
        if self.config.output_attentions == True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:, :num_joints, :]
        pred_vertices_sub = features[:, num_joints:-49, :]

        # learn camera parameters
        x = self.cam_param_fc(features[:, :-49, :])
        x = x.transpose(1, 2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1, 2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub.transpose(1, 2)
        pred_vertices = self.upsampling(temp_transpose)
        pred_vertices = pred_vertices.transpose(1, 2)

        if self.config.output_attentions == True:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices
