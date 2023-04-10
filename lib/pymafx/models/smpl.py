# This script is extended based on https://github.com/nkolot/SPIN/blob/master/models/smpl.py

import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from lib.pymafx.core import constants, path_config
from lib.smplx import SMPL as _SMPL
from lib.smplx import FLAMELayer, MANOLayer, SMPLXLayer
from lib.smplx.body_models import SMPLXOutput
from lib.smplx.lbs import (
    batch_rodrigues,
    blend_shapes,
    transform_mat,
    vertices2joints,
)

SMPL_MEAN_PARAMS = path_config.SMPL_MEAN_PARAMS
SMPL_MODEL_DIR = path_config.SMPL_MODEL_DIR


@dataclass
class ModelOutput(SMPLXOutput):
    smpl_joints: Optional[torch.Tensor] = None
    joints_J19: Optional[torch.Tensor] = None
    smplx_vertices: Optional[torch.Tensor] = None
    flame_vertices: Optional[torch.Tensor] = None
    lhand_vertices: Optional[torch.Tensor] = None
    rhand_vertices: Optional[torch.Tensor] = None
    lhand_joints: Optional[torch.Tensor] = None
    rhand_joints: Optional[torch.Tensor] = None
    face_joints: Optional[torch.Tensor] = None
    lfoot_joints: Optional[torch.Tensor] = None
    rfoot_joints: Optional[torch.Tensor] = None


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """
    def __init__(
        self,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
        *args,
        **kwargs
    ):
        super().__init__(
            create_betas=create_betas,
            create_global_orient=create_global_orient,
            create_body_pose=create_body_pose,
            create_transl=create_transl,
            *args,
            **kwargs
        )
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(path_config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32)
        )
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        # self.ModelOutput = namedtuple('ModelOutput_', ModelOutput._fields + ('smpl_joints', 'joints_J19',))
        # self.ModelOutput.__new__.__defaults__ = (None,) * len(self.ModelOutput._fields)

        tpose_joints = vertices2joints(self.J_regressor, self.v_template.unsqueeze(0))
        self.register_buffer('tpose_joints', tpose_joints)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super().forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        vertices = smpl_output.vertices
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        smpl_joints = smpl_output.joints[:, :24]
        joints = joints[:, self.joint_map, :]    # [B, 49, 3]
        joints_J24 = joints[:, -24:, :]
        joints_J19 = joints_J24[:, constants.J24_TO_J19, :]
        output = ModelOutput(
            vertices=vertices,
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            joints_J19=joints_J19,
            smpl_joints=smpl_joints,
            betas=smpl_output.betas,
            full_pose=smpl_output.full_pose
        )
        return output

    def get_global_rotation(
        self,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        **kwargs
    ):
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. It is expected to be in rotation matrix
                format. (default=None)
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            Returns
            -------
                output: Global rotation matrix
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [global_orient, body_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.eye(3, device=device,
                                      dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1,
                                                                           -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(
                batch_size, self.NUM_BODY_JOINTS, -1, -1
            ).contiguous()

        # Concatenate all pose vectors
        full_pose = torch.cat([
            global_orient.reshape(-1, 1, 3, 3),
            body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3)
        ],
                              dim=1)

        rot_mats = full_pose.view(batch_size, -1, 3, 3)

        # Get the joints
        # NxJx3 array
        # joints = vertices2joints(self.J_regressor, self.v_template.unsqueeze(0).expand(batch_size, -1, -1))
        # joints = torch.unsqueeze(joints, dim=-1)

        joints = self.tpose_joints.expand(batch_size, -1, -1).unsqueeze(-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]

        transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3),
                                       rel_joints.reshape(-1, 3,
                                                          1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, self.parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[self.parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        global_rotmat = transforms[:, :, :3, :3]

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        return global_rotmat, posed_joints


class SMPLX(SMPLXLayer):
    """ Extension of the official SMPLX implementation to support more functions """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_global_rotation(
        self,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        left_hand_pose: Optional[torch.Tensor] = None,
        right_hand_pose: Optional[torch.Tensor] = None,
        jaw_pose: Optional[torch.Tensor] = None,
        leye_pose: Optional[torch.Tensor] = None,
        reye_pose: Optional[torch.Tensor] = None,
        **kwargs
    ):
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. It is expected to be in rotation matrix
                format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                Expression coefficients.
                For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape BxJx3x3
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            left_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the left hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            right_hand_pose: torch.tensor, optional, shape Bx15x3x3
                If given, contains the pose of the right hand.
                It should be a tensor that contains joint rotations in
                rotation matrix format. (default=None)
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A data class that contains the posed vertices and joints
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.eye(3, device=device,
                                      dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1,
                                                                           -1).contiguous()
        if body_pose is None:
            body_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(
                batch_size, self.NUM_BODY_JOINTS, -1, -1
            ).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.eye(3, device=device,
                                       dtype=dtype).view(1, 1, 3, 3).expand(batch_size, 15, -1,
                                                                            -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.eye(3, device=device,
                                        dtype=dtype).view(1, 1, 3,
                                                          3).expand(batch_size, 15, -1,
                                                                    -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device,
                                 dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1,
                                                                      -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device,
                                  dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1,
                                                                       -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device,
                                  dtype=dtype).view(1, 1, 3, 3).expand(batch_size, -1, -1,
                                                                       -1).contiguous()

        # Concatenate all pose vectors
        full_pose = torch.cat([
            global_orient.reshape(-1, 1, 3, 3),
            body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3, 3),
            jaw_pose.reshape(-1, 1, 3, 3),
            leye_pose.reshape(-1, 1, 3, 3),
            reye_pose.reshape(-1, 1, 3, 3),
            left_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3),
            right_hand_pose.reshape(-1, self.NUM_HAND_JOINTS, 3, 3)
        ],
                              dim=1)

        rot_mats = full_pose.view(batch_size, -1, 3, 3)

        # Get the joints
        # NxJx3 array
        joints = vertices2joints(
            self.J_regressor,
            self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        )

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]

        transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3),
                                       rel_joints.reshape(-1, 3,
                                                          1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, self.parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[self.parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        global_rotmat = transforms[:, :, :3, :3]

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        return global_rotmat, posed_joints


class SMPLX_ALL(nn.Module):
    """ Extension of the official SMPLX implementation to support more joints """
    def __init__(self, batch_size=1, use_face_contour=True, all_gender=False, **kwargs):
        super().__init__()
        numBetas = 10
        self.use_face_contour = use_face_contour
        if all_gender:
            self.genders = ['male', 'female', 'neutral']
        else:
            self.genders = ['neutral']
        for gender in self.genders:
            assert gender in ['male', 'female', 'neutral']
        self.model_dict = nn.ModuleDict({
            gender: SMPLX(
                path_config.SMPL_MODEL_DIR,
                gender=gender,
                ext='npz',
                num_betas=numBetas,
                use_pca=False,
                batch_size=batch_size,
                use_face_contour=use_face_contour,
                num_pca_comps=45,
                **kwargs
            )
            for gender in self.genders
        })
        self.model_neutral = self.model_dict['neutral']
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(path_config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32)
        )
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        # smplx_to_smpl.pkl, file source: https://smpl-x.is.tue.mpg.de
        smplx_to_smpl = pickle.load(
            open(os.path.join(SMPL_MODEL_DIR, 'model_transfer/smplx_to_smpl.pkl'), 'rb')
        )
        self.register_buffer(
            'smplx2smpl', torch.tensor(smplx_to_smpl['matrix'][None], dtype=torch.float32)
        )

        smpl2limb_vert_faces = get_partial_smpl('smpl')
        self.smpl2lhand = torch.from_numpy(smpl2limb_vert_faces['lhand']['vids']).long()
        self.smpl2rhand = torch.from_numpy(smpl2limb_vert_faces['rhand']['vids']).long()

        # left and right hand joint mapping
        smplx2lhand_joints = [
            constants.SMPLX_JOINT_IDS['left_{}'.format(name)] for name in constants.HAND_NAMES
        ]
        smplx2rhand_joints = [
            constants.SMPLX_JOINT_IDS['right_{}'.format(name)] for name in constants.HAND_NAMES
        ]
        self.smplx2lh_joint_map = torch.tensor(smplx2lhand_joints, dtype=torch.long)
        self.smplx2rh_joint_map = torch.tensor(smplx2rhand_joints, dtype=torch.long)

        # left and right foot joint mapping
        smplx2lfoot_joints = [
            constants.SMPLX_JOINT_IDS['left_{}'.format(name)] for name in constants.FOOT_NAMES
        ]
        smplx2rfoot_joints = [
            constants.SMPLX_JOINT_IDS['right_{}'.format(name)] for name in constants.FOOT_NAMES
        ]
        self.smplx2lf_joint_map = torch.tensor(smplx2lfoot_joints, dtype=torch.long)
        self.smplx2rf_joint_map = torch.tensor(smplx2rfoot_joints, dtype=torch.long)

        for g in self.genders:
            J_template = torch.einsum(
                'ji,ik->jk', [self.model_dict[g].J_regressor[:24], self.model_dict[g].v_template]
            )
            J_dirs = torch.einsum(
                'ji,ikl->jkl', [self.model_dict[g].J_regressor[:24], self.model_dict[g].shapedirs]
            )

            self.register_buffer(f'{g}_J_template', J_template)
            self.register_buffer(f'{g}_J_dirs', J_dirs)

    def forward(self, *args, **kwargs):
        batch_size = kwargs['body_pose'].shape[0]
        kwargs['get_skin'] = True
        if 'pose2rot' not in kwargs:
            kwargs['pose2rot'] = True
        if 'gender' not in kwargs:
            kwargs['gender'] = 2 * torch.ones(batch_size).to(kwargs['body_pose'].device)

        # pose for 55 joints: 1, 21, 15, 15, 1, 1, 1
        pose_keys = [
            'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose',
            'leye_pose', 'reye_pose'
        ]
        param_keys = ['betas'] + pose_keys
        if kwargs['pose2rot']:
            for key in pose_keys:
                if key in kwargs:
                    # if key == 'left_hand_pose':
                    #     kwargs[key] += self.model_neutral.left_hand_mean
                    # elif key == 'right_hand_pose':
                    #     kwargs[key] += self.model_neutral.right_hand_mean
                    kwargs[key] = batch_rodrigues(kwargs[key].contiguous().view(-1, 3)).view([
                        batch_size, -1, 3, 3
                    ])
        if kwargs['body_pose'].shape[1] == 23:
            # remove hand pose in the body_pose
            kwargs['body_pose'] = kwargs['body_pose'][:, :21]
        gender_idx_list = []
        smplx_vertices, smplx_joints = [], []
        for gi, g in enumerate(['male', 'female', 'neutral']):
            gender_idx = ((kwargs['gender'] == gi).nonzero(as_tuple=True)[0])
            if len(gender_idx) == 0:
                continue
            gender_idx_list.extend([int(idx) for idx in gender_idx])
            gender_kwargs = {'get_skin': kwargs['get_skin'], 'pose2rot': kwargs['pose2rot']}
            gender_kwargs.update({k: kwargs[k][gender_idx] for k in param_keys if k in kwargs})
            gender_smplx_output = self.model_dict[g].forward(*args, **gender_kwargs)
            smplx_vertices.append(gender_smplx_output.vertices)
            smplx_joints.append(gender_smplx_output.joints)

        idx_rearrange = [gender_idx_list.index(i) for i in range(len(list(gender_idx_list)))]
        idx_rearrange = torch.tensor(idx_rearrange).long().to(kwargs['body_pose'].device)

        smplx_vertices = torch.cat(smplx_vertices)[idx_rearrange]
        smplx_joints = torch.cat(smplx_joints)[idx_rearrange]

        # constants.HAND_NAMES
        lhand_joints = smplx_joints[:, self.smplx2lh_joint_map]
        rhand_joints = smplx_joints[:, self.smplx2rh_joint_map]
        # constants.FACIAL_LANDMARKS
        face_joints = smplx_joints[:, -68:] if self.use_face_contour else smplx_joints[:, -51:]
        # constants.FOOT_NAMES
        lfoot_joints = smplx_joints[:, self.smplx2lf_joint_map]
        rfoot_joints = smplx_joints[:, self.smplx2rf_joint_map]

        smpl_vertices = torch.bmm(self.smplx2smpl.expand(batch_size, -1, -1), smplx_vertices)
        lhand_vertices = smpl_vertices[:, self.smpl2lhand]
        rhand_vertices = smpl_vertices[:, self.smpl2rhand]
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        smplx_j45 = smplx_joints[:, constants.SMPLX2SMPL_J45]
        joints = torch.cat([smplx_j45, extra_joints], dim=1)
        smpl_joints = smplx_j45[:, :24]
        joints = joints[:, self.joint_map, :]    # [B, 49, 3]
        joints_J24 = joints[:, -24:, :]
        joints_J19 = joints_J24[:, constants.J24_TO_J19, :]
        output = ModelOutput(
            vertices=smpl_vertices,
            smplx_vertices=smplx_vertices,
            lhand_vertices=lhand_vertices,
            rhand_vertices=rhand_vertices,
        # global_orient=smplx_output.global_orient,
        # body_pose=smplx_output.body_pose,
            joints=joints,
            joints_J19=joints_J19,
            smpl_joints=smpl_joints,
        # betas=smplx_output.betas,
        # full_pose=smplx_output.full_pose,
            lhand_joints=lhand_joints,
            rhand_joints=rhand_joints,
            lfoot_joints=lfoot_joints,
            rfoot_joints=rfoot_joints,
            face_joints=face_joints,
        )
        return output

    # def make_hand_regressor(self):
    #     # borrowed from https://github.com/mks0601/Hand4Whole_RELEASE/blob/main/common/utils/human_models.py
    #     regressor = self.model_neutral.J_regressor.numpy()
    #     vertex_num = self.model_neutral.J_regressor.shape[-1]
    #     lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],
    #                                         np.eye(vertex_num)[5361,None],
    #                                             regressor[[25,26,27],:],
    #                                             np.eye(vertex_num)[4933,None],
    #                                             regressor[[28,29,30],:],
    #                                             np.eye(vertex_num)[5058,None],
    #                                             regressor[[34,35,36],:],
    #                                             np.eye(vertex_num)[5169,None],
    #                                             regressor[[31,32,33],:],
    #                                             np.eye(vertex_num)[5286,None]))
    #     rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],
    #                                         np.eye(vertex_num)[8079,None],
    #                                             regressor[[40,41,42],:],
    #                                             np.eye(vertex_num)[7669,None],
    #                                             regressor[[43,44,45],:],
    #                                             np.eye(vertex_num)[7794,None],
    #                                             regressor[[49,50,51],:],
    #                                             np.eye(vertex_num)[7905,None],
    #                                             regressor[[46,47,48],:],
    #                                             np.eye(vertex_num)[8022,None]))
    #     return torch.from_numpy(lhand_regressor).float(), torch.from_numpy(rhand_regressor).float()

    def get_tpose(self, betas=None, gender=None):
        kwargs = {}
        if betas is None:
            betas = torch.zeros(1, 10).to(self.J_regressor_extra.device)
        kwargs['betas'] = betas

        batch_size = kwargs['betas'].shape[0]
        device = kwargs['betas'].device

        if gender is None:
            kwargs['gender'] = 2 * torch.ones(batch_size).to(device)
        else:
            kwargs['gender'] = gender

        param_keys = ['betas']

        gender_idx_list = []
        smplx_joints = []
        for gi, g in enumerate(['male', 'female', 'neutral']):
            gender_idx = ((kwargs['gender'] == gi).nonzero(as_tuple=True)[0])
            if len(gender_idx) == 0:
                continue
            gender_idx_list.extend([int(idx) for idx in gender_idx])
            gender_kwargs = {}
            gender_kwargs.update({k: kwargs[k][gender_idx] for k in param_keys if k in kwargs})

            J = getattr(self, f'{g}_J_template').unsqueeze(0) + blend_shapes(
                gender_kwargs['betas'], getattr(self, f'{g}_J_dirs')
            )

            smplx_joints.append(J)

        idx_rearrange = [gender_idx_list.index(i) for i in range(len(list(gender_idx_list)))]
        idx_rearrange = torch.tensor(idx_rearrange).long().to(device)

        smplx_joints = torch.cat(smplx_joints)[idx_rearrange]

        return smplx_joints


class MANO(MANOLayer):
    """ Extension of the official MANO implementation to support more joints """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if 'pose2rot' not in kwargs:
            kwargs['pose2rot'] = True
        pose_keys = ['global_orient', 'right_hand_pose']
        batch_size = kwargs['global_orient'].shape[0]
        if kwargs['pose2rot']:
            for key in pose_keys:
                if key in kwargs:
                    kwargs[key] = batch_rodrigues(kwargs[key].contiguous().view(-1, 3)).view([
                        batch_size, -1, 3, 3
                    ])
        kwargs['hand_pose'] = kwargs.pop('right_hand_pose')
        mano_output = super().forward(*args, **kwargs)
        th_verts = mano_output.vertices
        th_jtr = mano_output.joints
        # https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py#L248-L260
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        tips = th_verts[:, [745, 317, 445, 556, 673]]
        th_jtr = torch.cat([th_jtr, tips], 1)
        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:,
                        [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        output = ModelOutput(
            rhand_vertices=th_verts,
            rhand_joints=th_jtr,
        )
        return output


class FLAME(FLAMELayer):
    """ Extension of the official FLAME implementation to support more joints """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if 'pose2rot' not in kwargs:
            kwargs['pose2rot'] = True
        pose_keys = ['global_orient', 'jaw_pose', 'leye_pose', 'reye_pose']
        batch_size = kwargs['global_orient'].shape[0]
        if kwargs['pose2rot']:
            for key in pose_keys:
                if key in kwargs:
                    kwargs[key] = batch_rodrigues(kwargs[key].contiguous().view(-1, 3)).view([
                        batch_size, -1, 3, 3
                    ])
        flame_output = super().forward(*args, **kwargs)
        output = ModelOutput(
            flame_vertices=flame_output.vertices,
            face_joints=flame_output.joints[:, 5:],
        )
        return output


class SMPL_Family():
    def __init__(self, model_type='smpl', *args, **kwargs):
        if model_type == 'smpl':
            self.model = SMPL(model_path=SMPL_MODEL_DIR, *args, **kwargs)
        elif model_type == 'smplx':
            self.model = SMPLX_ALL(*args, **kwargs)
        elif model_type == 'mano':
            self.model = MANO(
                model_path=SMPL_MODEL_DIR, is_rhand=True, use_pca=False, *args, **kwargs
            )
        elif model_type == 'flame':
            self.model = FLAME(model_path=SMPL_MODEL_DIR, use_face_contour=True, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_tpose(self, *args, **kwargs):
        return self.model.get_tpose(*args, **kwargs)

    # def to(self, device):
    #     self.model.to(device)

    # def cuda(self, device=None):
    #     if device is None:
    #         self.model.cuda()
    #     else:
    #         self.model.cuda(device)


def get_smpl_faces():
    smpl = SMPL(model_path=SMPL_MODEL_DIR, batch_size=1)
    return smpl.faces


def get_smplx_faces():
    smplx = SMPLX(SMPL_MODEL_DIR, batch_size=1)
    return smplx.faces


def get_mano_faces(hand_type='right'):
    assert hand_type in ['right', 'left']
    is_rhand = True if hand_type == 'right' else False
    mano = MANO(SMPL_MODEL_DIR, batch_size=1, is_rhand=is_rhand)

    return mano.faces


def get_flame_faces():
    flame = FLAME(SMPL_MODEL_DIR, batch_size=1)

    return flame.faces


def get_model_faces(type='smpl'):
    if type == 'smpl':
        return get_smpl_faces()
    elif type == 'smplx':
        return get_smplx_faces()
    elif type == 'mano':
        return get_mano_faces()
    elif type == 'flame':
        return get_flame_faces()


def get_model_tpose(type='smpl'):
    if type == 'smpl':
        return get_smpl_tpose()
    elif type == 'smplx':
        return get_smplx_tpose()
    elif type == 'mano':
        return get_mano_tpose()
    elif type == 'flame':
        return get_flame_tpose()


def get_smpl_tpose():
    smpl = SMPL(
        create_betas=True,
        create_global_orient=True,
        create_body_pose=True,
        model_path=SMPL_MODEL_DIR,
        batch_size=1
    )
    vertices = smpl().vertices[0]
    return vertices.detach()


def get_smpl_tpose_joint():
    smpl = SMPL(
        create_betas=True,
        create_global_orient=True,
        create_body_pose=True,
        model_path=SMPL_MODEL_DIR,
        batch_size=1
    )
    tpose_joint = smpl().smpl_joints[0]
    return tpose_joint.detach()


def get_smplx_tpose():
    smplx = SMPLXLayer(SMPL_MODEL_DIR, batch_size=1)
    vertices = smplx().vertices[0]
    return vertices


def get_smplx_tpose_joint():
    smplx = SMPLXLayer(SMPL_MODEL_DIR, batch_size=1)
    tpose_joint = smplx().joints[0]
    return tpose_joint


def get_mano_tpose():
    mano = MANO(SMPL_MODEL_DIR, batch_size=1, is_rhand=True)
    vertices = mano(global_orient=torch.zeros(1, 3),
                    right_hand_pose=torch.zeros(1, 15 * 3)).rhand_vertices[0]
    return vertices


def get_flame_tpose():
    flame = FLAME(SMPL_MODEL_DIR, batch_size=1)
    vertices = flame(global_orient=torch.zeros(1, 3)).flame_vertices[0]
    return vertices


def get_part_joints(smpl_joints):
    batch_size = smpl_joints.shape[0]

    # part_joints = torch.zeros().to(smpl_joints.device)

    one_seg_pairs = [(0, 1), (0, 2), (0, 3), (3, 6), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16),
                     (14, 17)]
    two_seg_pairs = [(1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19), (18, 20), (19, 21)]

    one_seg_pairs.extend(two_seg_pairs)

    single_joints = [(10), (11), (15), (22), (23)]

    part_joints = []

    for j_p in one_seg_pairs:
        new_joint = torch.mean(smpl_joints[:, j_p], dim=1, keepdim=True)
        part_joints.append(new_joint)

    for j_p in single_joints:
        part_joints.append(smpl_joints[:, j_p:j_p + 1])

    part_joints = torch.cat(part_joints, dim=1)

    return part_joints


def get_partial_smpl(body_model='smpl', device=torch.device('cuda')):

    body_model_faces = get_model_faces(body_model)
    body_model_num_verts = len(get_model_tpose(body_model))

    part_vert_faces = {}

    for part in ['lhand', 'rhand', 'face', 'arm', 'forearm', 'larm', 'rarm', 'lwrist', 'rwrist']:
        part_vid_fname = '{}/{}_{}_vids.npz'.format(path_config.PARTIAL_MESH_DIR, body_model, part)
        if os.path.exists(part_vid_fname):
            part_vids = np.load(part_vid_fname)
            part_vert_faces[part] = {'vids': part_vids['vids'], 'faces': part_vids['faces']}
        else:
            if part in ['lhand', 'rhand']:
                with open(
                    os.path.join(SMPL_MODEL_DIR, 'model_transfer/MANO_SMPLX_vertex_ids.pkl'), 'rb'
                ) as json_file:
                    smplx_mano_id = pickle.load(json_file)
                with open(
                    os.path.join(SMPL_MODEL_DIR, 'model_transfer/smplx_to_smpl.pkl'), 'rb'
                ) as json_file:
                    smplx_smpl_id = pickle.load(json_file)

                smplx_tpose = get_smplx_tpose()
                smpl_tpose = np.matmul(smplx_smpl_id['matrix'], smplx_tpose)

                if part == 'lhand':
                    mano_vert = smplx_tpose[smplx_mano_id['left_hand']]
                elif part == 'rhand':
                    mano_vert = smplx_tpose[smplx_mano_id['right_hand']]

                smpl2mano_id = []
                for vert in mano_vert:
                    v_diff = smpl_tpose - vert
                    v_diff = torch.sum(v_diff * v_diff, dim=1)
                    v_closest = torch.argmin(v_diff)
                    smpl2mano_id.append(int(v_closest))

                smpl2mano_vids = np.array(smpl2mano_id).astype(np.long)
                mano_faces = get_mano_faces(hand_type='right' if part == 'rhand' else 'left'
                                           ).astype(np.long)

                np.savez(part_vid_fname, vids=smpl2mano_vids, faces=mano_faces)
                part_vert_faces[part] = {'vids': smpl2mano_vids, 'faces': mano_faces}

            elif part in ['face', 'arm', 'forearm', 'larm', 'rarm']:
                with open(
                    os.path.join(SMPL_MODEL_DIR, '{}_vert_segmentation.json'.format(body_model)),
                    'rb'
                ) as json_file:
                    smplx_part_id = json.load(json_file)

                # main_body_part = list(smplx_part_id.keys())
                # print('main_body_part', main_body_part)

                if part == 'face':
                    selected_body_part = ['head']
                elif part == 'arm':
                    selected_body_part = [
                        'rightHand',
                        'leftArm',
                        'leftShoulder',
                        'rightShoulder',
                        'rightArm',
                        'leftHandIndex1',
                        'rightHandIndex1',
                        'leftForeArm',
                        'rightForeArm',
                        'leftHand',
                    ]
                    # selected_body_part = ['rightHand', 'leftArm', 'rightArm', 'leftHandIndex1', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'leftHand',]
                elif part == 'forearm':
                    selected_body_part = [
                        'rightHand',
                        'leftHandIndex1',
                        'rightHandIndex1',
                        'leftForeArm',
                        'rightForeArm',
                        'leftHand',
                    ]
                elif part == 'arm_eval':
                    selected_body_part = ['leftArm', 'rightArm', 'leftForeArm', 'rightForeArm']
                elif part == 'larm':
                    # selected_body_part = ['leftArm', 'leftForeArm']
                    selected_body_part = ['leftForeArm']
                elif part == 'rarm':
                    # selected_body_part = ['rightArm', 'rightForeArm']
                    selected_body_part = ['rightForeArm']

                part_body_idx = []
                for k in selected_body_part:
                    part_body_idx.extend(smplx_part_id[k])

                part_body_fid = []
                for f_id, face in enumerate(body_model_faces):
                    if any(f in part_body_idx for f in face):
                        part_body_fid.append(f_id)

                smpl2head_vids = np.unique(body_model_faces[part_body_fid]).astype(np.long)

                mesh_vid_raw = np.arange(body_model_num_verts)
                head_vid_new = np.arange(len(smpl2head_vids))
                mesh_vid_raw[smpl2head_vids] = head_vid_new

                head_faces = body_model_faces[part_body_fid]
                head_faces = mesh_vid_raw[head_faces].astype(np.long)

                np.savez(part_vid_fname, vids=smpl2head_vids, faces=head_faces)
                part_vert_faces[part] = {'vids': smpl2head_vids, 'faces': head_faces}

            elif part in ['lwrist', 'rwrist']:

                if body_model == 'smplx':
                    body_model_verts = get_smplx_tpose()
                    tpose_joint = get_smplx_tpose_joint()
                elif body_model == 'smpl':
                    body_model_verts = get_smpl_tpose()
                    tpose_joint = get_smpl_tpose_joint()

                wrist_joint = tpose_joint[20] if part == 'lwrist' else tpose_joint[21]

                dist = 0.005
                wrist_vids = []
                for vid, vt in enumerate(body_model_verts):

                    v_j_dist = torch.sum((vt - wrist_joint)**2)

                    if v_j_dist < dist:
                        wrist_vids.append(vid)

                wrist_vids = np.array(wrist_vids)

                part_body_fid = []
                for f_id, face in enumerate(body_model_faces):
                    if any(f in wrist_vids for f in face):
                        part_body_fid.append(f_id)

                smpl2part_vids = np.unique(body_model_faces[part_body_fid]).astype(np.long)

                mesh_vid_raw = np.arange(body_model_num_verts)
                part_vid_new = np.arange(len(smpl2part_vids))
                mesh_vid_raw[smpl2part_vids] = part_vid_new

                part_faces = body_model_faces[part_body_fid]
                part_faces = mesh_vid_raw[part_faces].astype(np.long)

                np.savez(part_vid_fname, vids=smpl2part_vids, faces=part_faces)
                part_vert_faces[part] = {'vids': smpl2part_vids, 'faces': part_faces}

                # import trimesh
                # mesh = trimesh.Trimesh(vertices=body_model_verts[smpl2part_vids], faces=part_faces, process=False)
                # mesh.export(f'results/smplx_{part}.obj')

                # mesh = trimesh.Trimesh(vertices=body_model_verts, faces=body_model_faces, process=False)
                # mesh.export(f'results/smplx_model.obj')

    return part_vert_faces
