import torch
import torch.nn as nn
import numpy as np
from lib.pymafx.core import constants

from lib.common.config import cfg
from lib.pymafx.utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d, projection, rotation_matrix_to_angle_axis, compute_twist_rotation
from .maf_extractor import MAF_Extractor, Mesh_Sampler
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, get_partial_smpl, SMPL_Family
from lib.smplx.lbs import batch_rodrigues
from .res_module import IUV_predict_layer
from .hr_module import get_hrnet_encoder
from .pose_resnet import get_resnet_encoder
from lib.pymafx.utils.imutils import j2d_processing
from lib.pymafx.utils.cam_params import homo_vector
from .attention import get_att_block

import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


class Regressor(nn.Module):
    def __init__(
        self,
        feat_dim,
        smpl_mean_params,
        use_cam_feats=False,
        feat_dim_hand=0,
        feat_dim_face=0,
        bhf_names=['body'],
        smpl_models={}
    ):
        super().__init__()

        npose = 24 * 6
        shape_dim = 10
        cam_dim = 3
        hand_dim = 15 * 6
        face_dim = 3 * 6 + 10

        self.body_feat_dim = feat_dim

        self.smpl_mode = (cfg.MODEL.MESH_MODEL == 'smpl')
        self.smplx_mode = (cfg.MODEL.MESH_MODEL == 'smplx')
        self.use_cam_feats = use_cam_feats

        cam_feat_len = 4 if self.use_cam_feats else 0

        self.bhf_names = bhf_names
        self.hand_only_mode = (cfg.TRAIN.BHF_MODE == 'hand_only')
        self.face_only_mode = (cfg.TRAIN.BHF_MODE == 'face_only')
        self.body_hand_mode = (cfg.TRAIN.BHF_MODE == 'body_hand')
        self.full_body_mode = (cfg.TRAIN.BHF_MODE == 'full_body')

        # if self.use_cam_feats:
        #     assert cfg.MODEL.USE_IWP_CAM is False
        if 'body' in self.bhf_names:
            self.fc1 = nn.Linear(feat_dim + npose + cam_feat_len + shape_dim + cam_dim, 1024)
            self.drop1 = nn.Dropout()
            self.fc2 = nn.Linear(1024, 1024)
            self.drop2 = nn.Dropout()
            self.decpose = nn.Linear(1024, npose)
            self.decshape = nn.Linear(1024, 10)
            self.deccam = nn.Linear(1024, 3)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if not self.smpl_mode:
            if self.hand_only_mode:
                self.part_names = ['rhand']
            elif self.face_only_mode:
                self.part_names = ['face']
            elif self.body_hand_mode:
                self.part_names = ['lhand', 'rhand']
            elif self.full_body_mode:
                self.part_names = ['lhand', 'rhand', 'face']
            else:
                self.part_names = []

            if 'rhand' in self.part_names:
                # self.fc1_hand = nn.Linear(feat_dim_hand + hand_dim + rh_orient_dim + rh_shape_dim + rh_cam_dim, 1024)
                self.fc1_hand = nn.Linear(feat_dim_hand + hand_dim, 1024)
                self.drop1_hand = nn.Dropout()
                self.fc2_hand = nn.Linear(1024, 1024)
                self.drop2_hand = nn.Dropout()

                # self.declhand = nn.Linear(1024, 15*6)
                self.decrhand = nn.Linear(1024, 15 * 6)
                # nn.init.xavier_uniform_(self.declhand.weight, gain=0.01)
                nn.init.xavier_uniform_(self.decrhand.weight, gain=0.01)

                if cfg.MODEL.MESH_MODEL == 'mano' or cfg.MODEL.PyMAF.OPT_WRIST:
                    rh_cam_dim = 3
                    rh_orient_dim = 6
                    rh_shape_dim = 10
                    self.fc3_hand = nn.Linear(
                        1024 + rh_orient_dim + rh_shape_dim + rh_cam_dim, 1024
                    )
                    self.drop3_hand = nn.Dropout()

                    self.decshape_rhand = nn.Linear(1024, 10)
                    self.decorient_rhand = nn.Linear(1024, 6)
                    self.deccam_rhand = nn.Linear(1024, 3)
                    nn.init.xavier_uniform_(self.decshape_rhand.weight, gain=0.01)
                    nn.init.xavier_uniform_(self.decorient_rhand.weight, gain=0.01)
                    nn.init.xavier_uniform_(self.deccam_rhand.weight, gain=0.01)

            if 'face' in self.part_names:
                self.fc1_face = nn.Linear(feat_dim_face + face_dim, 1024)
                self.drop1_face = nn.Dropout()
                self.fc2_face = nn.Linear(1024, 1024)
                self.drop2_face = nn.Dropout()

                self.dechead = nn.Linear(1024, 3 * 6)
                self.decexp = nn.Linear(1024, 10)
                nn.init.xavier_uniform_(self.dechead.weight, gain=0.01)
                nn.init.xavier_uniform_(self.decexp.weight, gain=0.01)

                if cfg.MODEL.MESH_MODEL == 'flame':
                    rh_cam_dim = 3
                    rh_orient_dim = 6
                    rh_shape_dim = 10
                    self.fc3_face = nn.Linear(
                        1024 + rh_orient_dim + rh_shape_dim + rh_cam_dim, 1024
                    )
                    self.drop3_face = nn.Dropout()

                    self.decshape_face = nn.Linear(1024, 10)
                    self.decorient_face = nn.Linear(1024, 6)
                    self.deccam_face = nn.Linear(1024, 3)
                    nn.init.xavier_uniform_(self.decshape_face.weight, gain=0.01)
                    nn.init.xavier_uniform_(self.decorient_face.weight, gain=0.01)
                    nn.init.xavier_uniform_(self.deccam_face.weight, gain=0.01)

            if self.smplx_mode and cfg.MODEL.PyMAF.PRED_VIS_H:
                self.fc1_vis = nn.Linear(1024 + 1024 + 1024, 1024)
                self.drop1_vis = nn.Dropout()
                self.fc2_vis = nn.Linear(1024, 1024)
                self.drop2_vis = nn.Dropout()
                self.decvis = nn.Linear(1024, 2)
                nn.init.xavier_uniform_(self.decvis.weight, gain=0.01)

        if 'body' in smpl_models:
            self.smpl = smpl_models['body']
        if 'hand' in smpl_models:
            self.mano = smpl_models['hand']
        if 'face' in smpl_models:
            self.flame = smpl_models['face']

        if cfg.MODEL.PyMAF.OPT_WRIST:
            self.body_model = SMPL(model_path=SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_orient', init_pose[:, :6])

        self.flip_vector = torch.ones((1, 9), dtype=torch.float32)
        self.flip_vector[:, [1, 2, 3, 6]] *= -1
        self.flip_vector = self.flip_vector.reshape(1, 3, 3)

        if not self.smpl_mode:
            lhand_mean_rot6d = rotmat_to_rot6d(
                batch_rodrigues(self.smpl.model.model_neutral.left_hand_mean.view(-1, 3)).view(
                    [-1, 3, 3]
                )
            )
            rhand_mean_rot6d = rotmat_to_rot6d(
                batch_rodrigues(self.smpl.model.model_neutral.right_hand_mean.view(-1, 3)).view(
                    [-1, 3, 3]
                )
            )
            init_lhand = lhand_mean_rot6d.reshape(-1).unsqueeze(0)
            init_rhand = rhand_mean_rot6d.reshape(-1).unsqueeze(0)
            # init_hand = torch.cat([init_lhand, init_rhand]).unsqueeze(0)
            init_face = rotmat_to_rot6d(torch.stack([torch.eye(3)] * 3)).reshape(-1).unsqueeze(0)
            init_exp = torch.zeros(10).unsqueeze(0)

        if self.smplx_mode or 'hand' in bhf_names:
            # init_hand = torch.cat([init_lhand, init_rhand]).unsqueeze(0)
            self.register_buffer('init_lhand', init_lhand)
            self.register_buffer('init_rhand', init_rhand)
        if self.smplx_mode or 'face' in bhf_names:
            self.register_buffer('init_face', init_face)
            self.register_buffer('init_exp', init_exp)

    def forward(
        self,
        x=None,
        n_iter=1,
        J_regressor=None,
        rw_cam={},
        init_mode=False,
        global_iter=-1,
        **kwargs
    ):
        if x is not None:
            batch_size = x.shape[0]
        else:
            if 'xc_rhand' in kwargs:
                batch_size = kwargs['xc_rhand'].shape[0]
            elif 'xc_face' in kwargs:
                batch_size = kwargs['xc_face'].shape[0]

        if 'body' in self.bhf_names:
            if 'init_pose' not in kwargs:
                kwargs['init_pose'] = self.init_pose.expand(batch_size, -1)
            if 'init_shape' not in kwargs:
                kwargs['init_shape'] = self.init_shape.expand(batch_size, -1)
            if 'init_cam' not in kwargs:
                kwargs['init_cam'] = self.init_cam.expand(batch_size, -1)

            pred_cam = kwargs['init_cam']
            pred_pose = kwargs['init_pose']
            pred_shape = kwargs['init_shape']

        if self.full_body_mode or self.body_hand_mode:
            if cfg.MODEL.PyMAF.OPT_WRIST:
                pred_rotmat_body = rot6d_to_rotmat(
                    pred_pose.reshape(batch_size, -1, 6)
                )    # .view(batch_size, 24, 3, 3)
            if cfg.MODEL.PyMAF.PRED_VIS_H:
                pred_vis_hands = None

        # if self.full_body_mode or 'hand' in self.bhf_names:
        if self.smplx_mode or 'hand' in self.bhf_names:
            if 'init_lhand' not in kwargs:
                # kwargs['init_lhand'] = self.init_lhand.expand(batch_size, -1)
                # init with **right** hand pose
                kwargs['init_lhand'] = self.init_rhand.expand(batch_size, -1)
            if 'init_rhand' not in kwargs:
                kwargs['init_rhand'] = self.init_rhand.expand(batch_size, -1)

            pred_lhand, pred_rhand = kwargs['init_lhand'], kwargs['init_rhand']

            if cfg.MODEL.MESH_MODEL == 'mano' or cfg.MODEL.PyMAF.OPT_WRIST:
                if 'init_orient_rh' not in kwargs:
                    kwargs['init_orient_rh'] = self.init_orient.expand(batch_size, -1)
                if 'init_shape_rh' not in kwargs:
                    kwargs['init_shape_rh'] = self.init_shape.expand(batch_size, -1)
                if 'init_cam_rh' not in kwargs:
                    kwargs['init_cam_rh'] = self.init_cam.expand(batch_size, -1)
                pred_orient_rh = kwargs['init_orient_rh']
                pred_shape_rh = kwargs['init_shape_rh']
                pred_cam_rh = kwargs['init_cam_rh']
                if cfg.MODEL.PyMAF.OPT_WRIST:
                    if 'init_orient_lh' not in kwargs:
                        kwargs['init_orient_lh'] = self.init_orient.expand(batch_size, -1)
                    if 'init_shape_lh' not in kwargs:
                        kwargs['init_shape_lh'] = self.init_shape.expand(batch_size, -1)
                    if 'init_cam_lh' not in kwargs:
                        kwargs['init_cam_lh'] = self.init_cam.expand(batch_size, -1)
                    pred_orient_lh = kwargs['init_orient_lh']
                    pred_shape_lh = kwargs['init_shape_lh']
                    pred_cam_lh = kwargs['init_cam_lh']
                if cfg.MODEL.MESH_MODEL == 'mano':
                    pred_cam = torch.cat([pred_cam_rh[:, 0:1] * 10., pred_cam_rh[:, 1:]], dim=1)

        # if self.full_body_mode or 'face' in self.bhf_names:
        if self.smplx_mode or 'face' in self.bhf_names:
            if 'init_face' not in kwargs:
                kwargs['init_face'] = self.init_face.expand(batch_size, -1)
            if 'init_hand' not in kwargs:
                kwargs['init_exp'] = self.init_exp.expand(batch_size, -1)

            pred_face = kwargs['init_face']
            pred_exp = kwargs['init_exp']

            if cfg.MODEL.MESH_MODEL == 'flame' or cfg.MODEL.PyMAF.OPT_WRIST:
                if 'init_orient_fa' not in kwargs:
                    kwargs['init_orient_fa'] = self.init_orient.expand(batch_size, -1)
                pred_orient_fa = kwargs['init_orient_fa']
                if 'init_shape_fa' not in kwargs:
                    kwargs['init_shape_fa'] = self.init_shape.expand(batch_size, -1)
                if 'init_cam_fa' not in kwargs:
                    kwargs['init_cam_fa'] = self.init_cam.expand(batch_size, -1)
                pred_shape_fa = kwargs['init_shape_fa']
                pred_cam_fa = kwargs['init_cam_fa']
                if cfg.MODEL.MESH_MODEL == 'flame':
                    pred_cam = torch.cat([pred_cam_fa[:, 0:1] * 10., pred_cam_fa[:, 1:]], dim=1)

        if not init_mode:
            for i in range(n_iter):
                if 'body' in self.bhf_names:
                    xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
                    if self.use_cam_feats:
                        if cfg.MODEL.USE_IWP_CAM:
                            # for IWP camera, simply use pre-defined values
                            vfov = torch.ones((batch_size, 1)).to(xc) * 0.8
                            crop_ratio = torch.ones((batch_size, 1)).to(xc) * 0.3
                            crop_center = torch.ones((batch_size, 2)).to(xc) * 0.5
                        else:
                            vfov = rw_cam['vfov'][:, None]
                            crop_ratio = rw_cam['crop_ratio'][:, None]
                            crop_center = rw_cam['bbox_center'] / torch.cat(
                                [rw_cam['img_w'][:, None], rw_cam['img_h'][:, None]], 1
                            )
                        xc = torch.cat([xc, vfov, crop_ratio, crop_center], 1)

                    xc = self.fc1(xc)
                    xc = self.drop1(xc)
                    xc = self.fc2(xc)
                    xc = self.drop2(xc)

                    pred_cam = self.deccam(xc) + pred_cam
                    pred_pose = self.decpose(xc) + pred_pose
                    pred_shape = self.decshape(xc) + pred_shape

                if not self.smpl_mode:
                    if self.hand_only_mode:
                        xc_rhand = kwargs['xc_rhand']
                        xc_rhand = torch.cat([xc_rhand, pred_rhand], 1)
                    elif self.face_only_mode:
                        xc_face = kwargs['xc_face']
                        xc_face = torch.cat([xc_face, pred_face, pred_exp], 1)
                    elif self.body_hand_mode:
                        xc_lhand, xc_rhand = kwargs['xc_lhand'], kwargs['xc_rhand']
                        xc_lhand = torch.cat([xc_lhand, pred_lhand], 1)
                        xc_rhand = torch.cat([xc_rhand, pred_rhand], 1)
                    elif self.full_body_mode:
                        xc_lhand, xc_rhand, xc_face = kwargs['xc_lhand'], kwargs['xc_rhand'
                                                                                ], kwargs['xc_face']
                        xc_lhand = torch.cat([xc_lhand, pred_lhand], 1)
                        xc_rhand = torch.cat([xc_rhand, pred_rhand], 1)
                        xc_face = torch.cat([xc_face, pred_face, pred_exp], 1)

                    if 'lhand' in self.part_names:
                        xc_lhand = self.drop1_hand(self.fc1_hand(xc_lhand))
                        xc_lhand = self.drop2_hand(self.fc2_hand(xc_lhand))
                        pred_lhand = self.decrhand(xc_lhand) + pred_lhand

                        if cfg.MODEL.PyMAF.OPT_WRIST:
                            xc_lhand = torch.cat(
                                [xc_lhand, pred_shape_lh, pred_orient_lh, pred_cam_lh], 1
                            )
                            xc_lhand = self.drop3_hand(self.fc3_hand(xc_lhand))

                            pred_shape_lh = self.decshape_rhand(xc_lhand) + pred_shape_lh
                            pred_orient_lh = self.decorient_rhand(xc_lhand) + pred_orient_lh
                            pred_cam_lh = self.deccam_rhand(xc_lhand) + pred_cam_lh

                    if 'rhand' in self.part_names:
                        xc_rhand = self.drop1_hand(self.fc1_hand(xc_rhand))
                        xc_rhand = self.drop2_hand(self.fc2_hand(xc_rhand))
                        pred_rhand = self.decrhand(xc_rhand) + pred_rhand

                        if cfg.MODEL.MESH_MODEL == 'mano' or cfg.MODEL.PyMAF.OPT_WRIST:
                            xc_rhand = torch.cat(
                                [xc_rhand, pred_shape_rh, pred_orient_rh, pred_cam_rh], 1
                            )
                            xc_rhand = self.drop3_hand(self.fc3_hand(xc_rhand))

                            pred_shape_rh = self.decshape_rhand(xc_rhand) + pred_shape_rh
                            pred_orient_rh = self.decorient_rhand(xc_rhand) + pred_orient_rh
                            pred_cam_rh = self.deccam_rhand(xc_rhand) + pred_cam_rh

                            if cfg.MODEL.MESH_MODEL == 'mano':
                                pred_cam = torch.cat(
                                    [pred_cam_rh[:, 0:1] * 10., pred_cam_rh[:, 1:] / 10.], dim=1
                                )

                    if 'face' in self.part_names:
                        xc_face = self.drop1_face(self.fc1_face(xc_face))
                        xc_face = self.drop2_face(self.fc2_face(xc_face))
                        pred_face = self.dechead(xc_face) + pred_face
                        pred_exp = self.decexp(xc_face) + pred_exp

                        if cfg.MODEL.MESH_MODEL == 'flame':
                            xc_face = torch.cat(
                                [xc_face, pred_shape_fa, pred_orient_fa, pred_cam_fa], 1
                            )
                            xc_face = self.drop3_face(self.fc3_face(xc_face))

                            pred_shape_fa = self.decshape_face(xc_face) + pred_shape_fa
                            pred_orient_fa = self.decorient_face(xc_face) + pred_orient_fa
                            pred_cam_fa = self.deccam_face(xc_face) + pred_cam_fa

                            if cfg.MODEL.MESH_MODEL == 'flame':
                                pred_cam = torch.cat(
                                    [pred_cam_fa[:, 0:1] * 10., pred_cam_fa[:, 1:] / 10.], dim=1
                                )

                    if self.full_body_mode or self.body_hand_mode:
                        if cfg.MODEL.PyMAF.PRED_VIS_H:
                            xc_vis = torch.cat([xc, xc_lhand, xc_rhand], 1)

                            xc_vis = self.drop1_vis(self.fc1_vis(xc_vis))
                            xc_vis = self.drop2_vis(self.fc2_vis(xc_vis))
                            pred_vis_hands = self.decvis(xc_vis)

                            pred_vis_lhand = pred_vis_hands[:, 0] > cfg.MODEL.PyMAF.HAND_VIS_TH
                            pred_vis_rhand = pred_vis_hands[:, 1] > cfg.MODEL.PyMAF.HAND_VIS_TH

                        if cfg.MODEL.PyMAF.OPT_WRIST:

                            pred_rotmat_body = rot6d_to_rotmat(
                                pred_pose.reshape(batch_size, -1, 6)
                            )    # .view(batch_size, 24, 3, 3)
                            pred_lwrist = pred_rotmat_body[:, 20]
                            pred_rwrist = pred_rotmat_body[:, 21]

                            pred_gl_body, body_joints = self.body_model.get_global_rotation(
                                global_orient=pred_rotmat_body[:, 0:1],
                                body_pose=pred_rotmat_body[:, 1:]
                            )
                            pred_gl_lelbow = pred_gl_body[:, 18]
                            pred_gl_relbow = pred_gl_body[:, 19]

                            target_gl_lwrist = rot6d_to_rotmat(
                                pred_orient_lh.reshape(batch_size, -1, 6)
                            )
                            target_gl_lwrist *= self.flip_vector.to(target_gl_lwrist.device)
                            target_gl_rwrist = rot6d_to_rotmat(
                                pred_orient_rh.reshape(batch_size, -1, 6)
                            )

                            opt_lwrist = torch.bmm(pred_gl_lelbow.transpose(1, 2), target_gl_lwrist)
                            opt_rwrist = torch.bmm(pred_gl_relbow.transpose(1, 2), target_gl_rwrist)

                            if cfg.MODEL.PyMAF.ADAPT_INTEGR:
                                # if cfg.MODEL.PyMAF.ADAPT_INTEGR and global_iter == (cfg.MODEL.PyMAF.N_ITER - 1):
                                tpose_joints = self.smpl.get_tpose(betas=pred_shape)
                                lelbow_twist_axis = nn.functional.normalize(
                                    tpose_joints[:, 20] - tpose_joints[:, 18], dim=1
                                )
                                relbow_twist_axis = nn.functional.normalize(
                                    tpose_joints[:, 21] - tpose_joints[:, 19], dim=1
                                )

                                lelbow_twist, lelbow_twist_angle = compute_twist_rotation(
                                    opt_lwrist, lelbow_twist_axis
                                )
                                relbow_twist, relbow_twist_angle = compute_twist_rotation(
                                    opt_rwrist, relbow_twist_axis
                                )

                                min_angle = -0.4 * float(np.pi)
                                max_angle = 0.4 * float(np.pi)

                                lelbow_twist_angle[lelbow_twist_angle == torch.
                                                   clamp(lelbow_twist_angle, min_angle, max_angle)
                                                  ] = 0
                                relbow_twist_angle[relbow_twist_angle == torch.
                                                   clamp(relbow_twist_angle, min_angle, max_angle)
                                                  ] = 0
                                lelbow_twist_angle[lelbow_twist_angle > max_angle] -= max_angle
                                lelbow_twist_angle[lelbow_twist_angle < min_angle] -= min_angle
                                relbow_twist_angle[relbow_twist_angle > max_angle] -= max_angle
                                relbow_twist_angle[relbow_twist_angle < min_angle] -= min_angle

                                lelbow_twist = batch_rodrigues(
                                    lelbow_twist_axis * lelbow_twist_angle
                                )
                                relbow_twist = batch_rodrigues(
                                    relbow_twist_axis * relbow_twist_angle
                                )

                                opt_lwrist = torch.bmm(lelbow_twist.transpose(1, 2), opt_lwrist)
                                opt_rwrist = torch.bmm(relbow_twist.transpose(1, 2), opt_rwrist)

                                # left elbow: 18
                                opt_lelbow = torch.bmm(pred_rotmat_body[:, 18], lelbow_twist)
                                # right elbow: 19
                                opt_relbow = torch.bmm(pred_rotmat_body[:, 19], relbow_twist)

                                if cfg.MODEL.PyMAF.PRED_VIS_H and global_iter == (
                                    cfg.MODEL.PyMAF.N_ITER - 1
                                ):
                                    opt_lwrist_filtered = [
                                        opt_lwrist[_i]
                                        if pred_vis_lhand[_i] else pred_rotmat_body[_i, 20]
                                        for _i in range(batch_size)
                                    ]
                                    opt_rwrist_filtered = [
                                        opt_rwrist[_i]
                                        if pred_vis_rhand[_i] else pred_rotmat_body[_i, 21]
                                        for _i in range(batch_size)
                                    ]
                                    opt_lelbow_filtered = [
                                        opt_lelbow[_i]
                                        if pred_vis_lhand[_i] else pred_rotmat_body[_i, 18]
                                        for _i in range(batch_size)
                                    ]
                                    opt_relbow_filtered = [
                                        opt_relbow[_i]
                                        if pred_vis_rhand[_i] else pred_rotmat_body[_i, 19]
                                        for _i in range(batch_size)
                                    ]

                                    opt_lwrist = torch.stack(opt_lwrist_filtered)
                                    opt_rwrist = torch.stack(opt_rwrist_filtered)
                                    opt_lelbow = torch.stack(opt_lelbow_filtered)
                                    opt_relbow = torch.stack(opt_relbow_filtered)

                                pred_rotmat_body = torch.cat(
                                    [
                                        pred_rotmat_body[:, :18],
                                        opt_lelbow.unsqueeze(1),
                                        opt_relbow.unsqueeze(1),
                                        opt_lwrist.unsqueeze(1),
                                        opt_rwrist.unsqueeze(1), pred_rotmat_body[:, 22:]
                                    ], 1
                                )
                            else:
                                if cfg.MODEL.PyMAF.PRED_VIS_H and global_iter == (
                                    cfg.MODEL.PyMAF.N_ITER - 1
                                ):
                                    opt_lwrist_filtered = [
                                        opt_lwrist[_i]
                                        if pred_vis_lhand[_i] else pred_rotmat_body[_i, 20]
                                        for _i in range(batch_size)
                                    ]
                                    opt_rwrist_filtered = [
                                        opt_rwrist[_i]
                                        if pred_vis_rhand[_i] else pred_rotmat_body[_i, 21]
                                        for _i in range(batch_size)
                                    ]

                                    opt_lwrist = torch.stack(opt_lwrist_filtered)
                                    opt_rwrist = torch.stack(opt_rwrist_filtered)

                                pred_rotmat_body = torch.cat(
                                    [
                                        pred_rotmat_body[:, :20],
                                        opt_lwrist.unsqueeze(1),
                                        opt_rwrist.unsqueeze(1), pred_rotmat_body[:, 22:]
                                    ], 1
                                )

        if self.hand_only_mode:
            pred_rotmat_rh = rot6d_to_rotmat(
                torch.cat([pred_orient_rh, pred_rhand], dim=1).reshape(batch_size, -1, 6)
            )    # .view(batch_size, 16, 3, 3)
            assert pred_rotmat_rh.shape[1] == 1 + 15
        elif self.face_only_mode:
            pred_rotmat_fa = rot6d_to_rotmat(
                torch.cat([pred_orient_fa, pred_face], dim=1).reshape(batch_size, -1, 6)
            )    # .view(batch_size, 16, 3, 3)
            assert pred_rotmat_fa.shape[1] == 1 + 3
        elif self.full_body_mode or self.body_hand_mode:
            if cfg.MODEL.PyMAF.OPT_WRIST:
                pred_rotmat = pred_rotmat_body
            else:
                pred_rotmat = rot6d_to_rotmat(
                    pred_pose.reshape(batch_size, -1, 6)
                )    # .view(batch_size, 24, 3, 3)
            assert pred_rotmat.shape[1] == 24
        else:
            pred_rotmat = rot6d_to_rotmat(
                pred_pose.reshape(batch_size, -1, 6)
            )    # .view(batch_size, 24, 3, 3)
            assert pred_rotmat.shape[1] == 24

        # if self.full_body_mode:
        if self.smplx_mode:
            if cfg.MODEL.PyMAF.PRED_VIS_H and global_iter == (cfg.MODEL.PyMAF.N_ITER - 1):
                pred_lhand_filtered = [
                    pred_lhand[_i] if pred_vis_lhand[_i] else self.init_rhand[0]
                    for _i in range(batch_size)
                ]
                pred_rhand_filtered = [
                    pred_rhand[_i] if pred_vis_rhand[_i] else self.init_rhand[0]
                    for _i in range(batch_size)
                ]
                pred_lhand_filtered = torch.stack(pred_lhand_filtered)
                pred_rhand_filtered = torch.stack(pred_rhand_filtered)
                pred_hf6d = torch.cat([pred_lhand_filtered, pred_rhand_filtered, pred_face],
                                      dim=1).reshape(batch_size, -1, 6)
            else:
                pred_hf6d = torch.cat([pred_lhand, pred_rhand, pred_face],
                                      dim=1).reshape(batch_size, -1, 6)
            pred_hfrotmat = rot6d_to_rotmat(pred_hf6d)
            assert pred_hfrotmat.shape[1] == (15 * 2 + 3)

            # flip left hand pose
            pred_lhand_rotmat = pred_hfrotmat[:, :15] * self.flip_vector.to(pred_hfrotmat.device
                                                                           ).unsqueeze(0)
            pred_rhand_rotmat = pred_hfrotmat[:, 15:30]
            pred_face_rotmat = pred_hfrotmat[:, 30:]

        if self.hand_only_mode:
            pred_output = self.mano(
                betas=pred_shape_rh,
                right_hand_pose=pred_rotmat_rh[:, 1:],
                global_orient=pred_rotmat_rh[:, 0].unsqueeze(1),
                pose2rot=False,
            )
        elif self.face_only_mode:
            pred_output = self.flame(
                betas=pred_shape_fa,
                global_orient=pred_rotmat_fa[:, 0].unsqueeze(1),
                jaw_pose=pred_rotmat_fa[:, 1:2],
                leye_pose=pred_rotmat_fa[:, 2:3],
                reye_pose=pred_rotmat_fa[:, 3:4],
                expression=pred_exp,
                pose2rot=False,
            )
        else:
            smplx_kwargs = {}
            # if self.full_body_mode:
            if self.smplx_mode:
                smplx_kwargs['left_hand_pose'] = pred_lhand_rotmat
                smplx_kwargs['right_hand_pose'] = pred_rhand_rotmat
                smplx_kwargs['jaw_pose'] = pred_face_rotmat[:, 0:1]
                smplx_kwargs['leye_pose'] = pred_face_rotmat[:, 1:2]
                smplx_kwargs['reye_pose'] = pred_face_rotmat[:, 2:3]
                smplx_kwargs['expression'] = pred_exp

            pred_output = self.smpl(
                betas=pred_shape,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
                **smplx_kwargs,
            )

            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

        if self.hand_only_mode:
            pred_joints_full = pred_output.rhand_joints
        elif self.face_only_mode:
            pred_joints_full = pred_output.face_joints
        elif self.smplx_mode:
            pred_joints_full = torch.cat(
                [
                    pred_joints, pred_output.lhand_joints, pred_output.rhand_joints,
                    pred_output.face_joints, pred_output.lfoot_joints, pred_output.rfoot_joints
                ],
                dim=1
            )
        else:
            pred_joints_full = pred_joints
        pred_keypoints_2d = projection(
            pred_joints_full, {
                **rw_cam, 'cam_sxy': pred_cam
            }, iwp_mode=cfg.MODEL.USE_IWP_CAM
        )
        if cfg.MODEL.USE_IWP_CAM:
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
        else:
            pred_keypoints_2d = j2d_processing(pred_keypoints_2d, rw_cam['kps_transf'])

        len_b_kp = len(constants.JOINT_NAMES)
        output = {}
        if self.smpl_mode or self.smplx_mode:
            if J_regressor is not None:
                kp_3d = torch.matmul(J_regressor, pred_vertices)
                pred_pelvis = kp_3d[:, [0], :].clone()
                kp_3d = kp_3d[:, constants.H36M_TO_J14, :]
                kp_3d = kp_3d - pred_pelvis
            else:
                kp_3d = pred_joints
            pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
            output.update(
                {
                    'theta': torch.cat([pred_cam, pred_shape, pose], dim=1),
                    'verts': pred_vertices,
                    'kp_2d': pred_keypoints_2d[:, :len_b_kp],
                    'kp_3d': kp_3d,
                    'pred_joints': pred_joints,
                    'smpl_kp_3d': pred_output.smpl_joints,
                    'rotmat': pred_rotmat,
                    'pred_cam': pred_cam,
                    'pred_shape': pred_shape,
                    'pred_pose': pred_pose,
                }
            )
            # if self.full_body_mode:
            if self.smplx_mode:
                # assert pred_keypoints_2d.shape[1] == 144
                len_h_kp = len(constants.HAND_NAMES)
                len_f_kp = len(constants.FACIAL_LANDMARKS)
                len_feet_kp = 2 * len(constants.FOOT_NAMES)
                output.update(
                    {
                        'smplx_verts':
                            pred_output.smplx_vertices if cfg.MODEL.EVAL_MODE else None,
                        'pred_lhand':
                            pred_lhand,
                        'pred_rhand':
                            pred_rhand,
                        'pred_face':
                            pred_face,
                        'pred_exp':
                            pred_exp,
                        'verts_lh':
                            pred_output.lhand_vertices,
                        'verts_rh':
                            pred_output.rhand_vertices,
                # 'pred_arm_rotmat': pred_arm_rotmat,
                # 'pred_hfrotmat': pred_hfrotmat,
                        'pred_lhand_rotmat':
                            pred_lhand_rotmat,
                        'pred_rhand_rotmat':
                            pred_rhand_rotmat,
                        'pred_face_rotmat':
                            pred_face_rotmat,
                        'pred_lhand_kp3d':
                            pred_output.lhand_joints,
                        'pred_rhand_kp3d':
                            pred_output.rhand_joints,
                        'pred_face_kp3d':
                            pred_output.face_joints,
                        'pred_lhand_kp2d':
                            pred_keypoints_2d[:, len_b_kp:len_b_kp + len_h_kp],
                        'pred_rhand_kp2d':
                            pred_keypoints_2d[:, len_b_kp + len_h_kp:len_b_kp + len_h_kp * 2],
                        'pred_face_kp2d':
                            pred_keypoints_2d[:, len_b_kp + len_h_kp * 2:len_b_kp + len_h_kp * 2 +
                                              len_f_kp],
                        'pred_feet_kp2d':
                            pred_keypoints_2d[:, len_b_kp + len_h_kp * 2 + len_f_kp:len_b_kp +
                                              len_h_kp * 2 + len_f_kp + len_feet_kp],
                    }
                )
                if cfg.MODEL.PyMAF.OPT_WRIST:
                    output.update(
                        {
                            'pred_orient_lh': pred_orient_lh,
                            'pred_shape_lh': pred_shape_lh,
                            'pred_orient_rh': pred_orient_rh,
                            'pred_shape_rh': pred_shape_rh,
                            'pred_cam_fa': pred_cam_fa,
                            'pred_cam_lh': pred_cam_lh,
                            'pred_cam_rh': pred_cam_rh,
                        }
                    )
                if cfg.MODEL.PyMAF.PRED_VIS_H:
                    output.update({'pred_vis_hands': pred_vis_hands})
        elif self.hand_only_mode:
            # hand mesh out
            assert pred_keypoints_2d.shape[1] == 21
            output.update(
                {
                    'theta': pred_cam,
                    'pred_cam': pred_cam,
                    'pred_rhand': pred_rhand,
                    'pred_rhand_rotmat': pred_rotmat_rh[:, 1:],
                    'pred_orient_rh': pred_orient_rh,
                    'pred_orient_rh_rotmat': pred_rotmat_rh[:, 0],
                    'verts_rh': pred_output.rhand_vertices,
                    'pred_cam_rh': pred_cam_rh,
                    'pred_shape_rh': pred_shape_rh,
                    'pred_rhand_kp3d': pred_output.rhand_joints,
                    'pred_rhand_kp2d': pred_keypoints_2d,
                }
            )
        elif self.face_only_mode:
            # face mesh out
            assert pred_keypoints_2d.shape[1] == 68
            output.update(
                {
                    'theta': pred_cam,
                    'pred_cam': pred_cam,
                    'pred_face': pred_face,
                    'pred_exp': pred_exp,
                    'pred_face_rotmat': pred_rotmat_fa[:, 1:],
                    'pred_orient_fa': pred_orient_fa,
                    'pred_orient_fa_rotmat': pred_rotmat_fa[:, 0],
                    'verts_fa': pred_output.flame_vertices,
                    'pred_cam_fa': pred_cam_fa,
                    'pred_shape_fa': pred_shape_fa,
                    'pred_face_kp3d': pred_output.face_joints,
                    'pred_face_kp2d': pred_keypoints_2d,
                }
            )
        return output


def get_attention_modules(
    module_keys, img_feature_dim_list, hidden_feat_dim, n_iter, num_attention_heads=1
):

    align_attention = nn.ModuleDict()
    for k in module_keys:
        align_attention[k] = nn.ModuleList()
        for i in range(n_iter):
            align_attention[k].append(
                get_att_block(
                    img_feature_dim=img_feature_dim_list[k][i],
                    hidden_feat_dim=hidden_feat_dim,
                    num_attention_heads=num_attention_heads
                )
            )

    return align_attention


def get_fusion_modules(module_keys, ma_feat_dim, grid_feat_dim, n_iter, out_feat_len):

    feat_fusion = nn.ModuleDict()
    for k in module_keys:
        feat_fusion[k] = nn.ModuleList()
        for i in range(n_iter):
            feat_fusion[k].append(nn.Linear(grid_feat_dim + ma_feat_dim[k], out_feat_len[k]))

    return feat_fusion


class PyMAF(nn.Module):
    """ PyMAF based Regression Network for Human Mesh Recovery / Full-body Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images, arXiv:2207.06400, 2022
    """
    def __init__(
        self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True, device=torch.device('cuda')
    ):
        super().__init__()

        self.device = device

        self.smpl_mode = (cfg.MODEL.MESH_MODEL == 'smpl')
        self.smplx_mode = (cfg.MODEL.MESH_MODEL == 'smplx')

        assert cfg.TRAIN.BHF_MODE in [
            'body_only', 'hand_only', 'face_only', 'body_hand', 'full_body'
        ]
        self.hand_only_mode = (cfg.TRAIN.BHF_MODE == 'hand_only')
        self.face_only_mode = (cfg.TRAIN.BHF_MODE == 'face_only')
        self.body_hand_mode = (cfg.TRAIN.BHF_MODE == 'body_hand')
        self.full_body_mode = (cfg.TRAIN.BHF_MODE == 'full_body')

        bhf_names = []
        if cfg.TRAIN.BHF_MODE in ['body_only', 'body_hand', 'full_body']:
            bhf_names.append('body')
        if cfg.TRAIN.BHF_MODE in ['hand_only', 'body_hand', 'full_body']:
            bhf_names.append('hand')
        if cfg.TRAIN.BHF_MODE in ['face_only', 'full_body']:
            bhf_names.append('face')
        self.bhf_names = bhf_names

        self.part_module_names = {'body': {}, 'hand': {}, 'face': {}, 'link': {}}

        # the limb parts need to be handled
        if self.hand_only_mode:
            self.part_names = ['rhand']
        elif self.face_only_mode:
            self.part_names = ['face']
        elif self.body_hand_mode:
            self.part_names = ['lhand', 'rhand']
        elif self.full_body_mode:
            self.part_names = ['lhand', 'rhand', 'face']
        else:
            self.part_names = []

        # joint index info
        if not self.smpl_mode:
            h_root_idx = constants.HAND_NAMES.index('wrist')
            h_idx = constants.HAND_NAMES.index('middle1')
            f_idx = constants.FACIAL_LANDMARKS.index('nose_middle')
            self.hf_center_idx = {'lhand': h_idx, 'rhand': h_idx, 'face': f_idx}
            self.hf_root_idx = {'lhand': h_root_idx, 'rhand': h_root_idx, 'face': f_idx}

            lh_idx_coco = constants.COCO_KEYPOINTS.index('left_wrist')
            rh_idx_coco = constants.COCO_KEYPOINTS.index('right_wrist')
            f_idx_coco = constants.COCO_KEYPOINTS.index('nose')
            self.hf_root_idx_coco = {'lhand': lh_idx_coco, 'rhand': rh_idx_coco, 'face': f_idx_coco}

        # create parametric mesh models
        self.smpl_family = {}
        if self.hand_only_mode and cfg.MODEL.MESH_MODEL == 'mano':
            self.smpl_family['hand'] = SMPL_Family(model_type='mano')
            self.smpl_family['body'] = SMPL_Family(model_type='smplx')
        elif self.face_only_mode and cfg.MODEL.MESH_MODEL == 'flame':
            self.smpl_family['face'] = SMPL_Family(model_type='flame')
            self.smpl_family['body'] = SMPL_Family(model_type='smplx')
        else:
            self.smpl_family['body'] = SMPL_Family(
                model_type=cfg.MODEL.MESH_MODEL, all_gender=cfg.MODEL.ALL_GENDER
            )

        self.init_mesh_output = None
        self.batch_size = 1

        self.encoders = nn.ModuleDict()
        self.global_mode = not cfg.MODEL.PyMAF.MAF_ON

        # build encoders
        global_feat_dim = 2048
        bhf_ma_feat_dim = {}
        # encoder for the body part
        if 'body' in bhf_names:
            # if self.smplx_mode or 'hr' in cfg.MODEL.PyMAF.BACKBONE:
            if cfg.MODEL.PyMAF.BACKBONE == 'res50':
                body_encoder = get_resnet_encoder(
                    cfg, init_weight=(not cfg.MODEL.EVAL_MODE), global_mode=self.global_mode
                )
                body_sfeat_dim = list(cfg.POSE_RES_MODEL.EXTRA.NUM_DECONV_FILTERS)
            elif cfg.MODEL.PyMAF.BACKBONE == 'hr48':
                body_encoder = get_hrnet_encoder(
                    cfg, init_weight=(not cfg.MODEL.EVAL_MODE), global_mode=self.global_mode
                )
                body_sfeat_dim = list(cfg.HR_MODEL.EXTRA.STAGE4.NUM_CHANNELS)
                body_sfeat_dim.reverse()
                body_sfeat_dim = body_sfeat_dim[1:]
            else:
                raise NotImplementedError
            self.encoders['body'] = body_encoder
            self.part_module_names['body'].update({'encoders.body': self.encoders['body']})

            self.mesh_sampler = Mesh_Sampler(type='smpl')
            self.part_module_names['body'].update({'mesh_sampler': self.mesh_sampler})

            if not cfg.MODEL.PyMAF.GRID_FEAT:
                ma_feat_dim = self.mesh_sampler.Dmap.shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]
            else:
                ma_feat_dim = 0
            bhf_ma_feat_dim['body'] = ma_feat_dim

            dp_feat_dim = body_sfeat_dim[-1]
            self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
            if cfg.MODEL.PyMAF.AUX_SUPV_ON:
                assert cfg.MODEL.PyMAF.MAF_ON
                self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)
                self.part_module_names['body'].update({'dp_head': self.dp_head})

        # encoders for the hand / face parts
        if 'hand' in self.bhf_names or 'face' in self.bhf_names:
            for hf in ['hand', 'face']:
                if hf in bhf_names:
                    if cfg.MODEL.PyMAF.HF_BACKBONE == 'res50':
                        self.encoders[hf] = get_resnet_encoder(
                            cfg,
                            init_weight=(not cfg.MODEL.EVAL_MODE),
                            global_mode=self.global_mode
                        )
                        self.part_module_names[hf].update({f'encoders.{hf}': self.encoders[hf]})
                        hf_sfeat_dim = list(cfg.POSE_RES_MODEL.EXTRA.NUM_DECONV_FILTERS)
                    else:
                        raise NotImplementedError

            if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                assert cfg.MODEL.PyMAF.MAF_ON
                self.dp_head_hf = nn.ModuleDict()
                if 'hand' in bhf_names:
                    self.dp_head_hf['hand'] = IUV_predict_layer(
                        feat_dim=hf_sfeat_dim[-1], mode='pncc'
                    )
                    self.part_module_names['hand'].update(
                        {'dp_head_hf.hand': self.dp_head_hf['hand']}
                    )
                if 'face' in bhf_names:
                    self.dp_head_hf['face'] = IUV_predict_layer(
                        feat_dim=hf_sfeat_dim[-1], mode='pncc'
                    )
                    self.part_module_names['face'].update(
                        {'dp_head_hf.face': self.dp_head_hf['face']}
                    )

            smpl2limb_vert_faces = get_partial_smpl()

            self.smpl2lhand = torch.from_numpy(smpl2limb_vert_faces['lhand']['vids']).long()
            self.smpl2rhand = torch.from_numpy(smpl2limb_vert_faces['rhand']['vids']).long()

        # grid points for grid feature extraction
        grid_size = 21
        xv, yv = torch.meshgrid(
            [torch.linspace(-1, 1, grid_size),
             torch.linspace(-1, 1, grid_size)]
        )
        grid_points = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('grid_points', grid_points)
        grid_feat_dim = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        # the fusion of grid and mesh-aligned features
        self.fuse_grid_align = cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT or cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC
        assert not (cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT and cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC)

        if self.fuse_grid_align:
            self.att_starts = cfg.MODEL.PyMAF.GRID_ALIGN.ATT_STARTS
            n_iter_att = cfg.MODEL.PyMAF.N_ITER - self.att_starts
            att_feat_dim_idx = -cfg.MODEL.PyMAF.GRID_ALIGN.ATT_FEAT_IDX
            num_att_heads = cfg.MODEL.PyMAF.GRID_ALIGN.ATT_HEAD
            hidden_feat_dim = cfg.MODEL.PyMAF.MLP_DIM[att_feat_dim_idx]
            bhf_att_feat_dim = {'body': 2048}

        if 'hand' in self.bhf_names:
            self.mano_sampler = Mesh_Sampler(type='mano', level=1)
            self.mano_ds_len = self.mano_sampler.Dmap.shape[0]
            self.part_module_names['hand'].update({'mano_sampler': self.mano_sampler})

            bhf_ma_feat_dim.update({'hand': self.mano_ds_len * cfg.MODEL.PyMAF.HF_MLP_DIM[-1]})

            if self.fuse_grid_align:
                bhf_att_feat_dim.update({'hand': 1024})

        if 'face' in self.bhf_names:
            bhf_ma_feat_dim.update(
                {'face': len(constants.FACIAL_LANDMARKS) * cfg.MODEL.PyMAF.HF_MLP_DIM[-1]}
            )
            if self.fuse_grid_align:
                bhf_att_feat_dim.update({'face': 1024})

        # spatial alignment attention
        if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
            hfimg_feat_dim_list = {}
            if 'body' in bhf_names:
                hfimg_feat_dim_list['body'] = body_sfeat_dim[-n_iter_att:]

            if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                if 'hand' in bhf_names:
                    hfimg_feat_dim_list['hand'] = hf_sfeat_dim[-n_iter_att:]
                if 'face' in bhf_names:
                    hfimg_feat_dim_list['face'] = hf_sfeat_dim[-n_iter_att:]

            self.align_attention = get_attention_modules(
                bhf_names,
                hfimg_feat_dim_list,
                hidden_feat_dim,
                n_iter=n_iter_att,
                num_attention_heads=num_att_heads
            )

            for part in bhf_names:
                self.part_module_names[part].update(
                    {f'align_attention.{part}': self.align_attention[part]}
                )

        if self.fuse_grid_align:
            self.att_feat_reduce = get_fusion_modules(
                bhf_names,
                bhf_ma_feat_dim,
                grid_feat_dim,
                n_iter=n_iter_att,
                out_feat_len=bhf_att_feat_dim
            )
            for part in bhf_names:
                self.part_module_names[part].update(
                    {f'att_feat_reduce.{part}': self.att_feat_reduce[part]}
                )

        # build regressor for parameter prediction
        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            ref_infeat_dim = 0
            if 'body' in self.bhf_names:
                if cfg.MODEL.PyMAF.MAF_ON:
                    if self.fuse_grid_align:
                        if i >= self.att_starts:
                            ref_infeat_dim = bhf_att_feat_dim['body']
                        elif i == 0 or cfg.MODEL.PyMAF.GRID_FEAT:
                            ref_infeat_dim = grid_feat_dim
                        else:
                            ref_infeat_dim = ma_feat_dim
                    else:
                        if i == 0 or cfg.MODEL.PyMAF.GRID_FEAT:
                            ref_infeat_dim = grid_feat_dim
                        else:
                            ref_infeat_dim = ma_feat_dim
                else:
                    ref_infeat_dim = global_feat_dim

            if self.smpl_mode:
                self.regressor.append(
                    Regressor(
                        feat_dim=ref_infeat_dim,
                        smpl_mean_params=smpl_mean_params,
                        use_cam_feats=cfg.MODEL.PyMAF.USE_CAM_FEAT,
                        smpl_models=self.smpl_family
                    )
                )
            else:
                if cfg.MODEL.PyMAF.MAF_ON:
                    if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                        if i == 0:
                            feat_dim_hand = grid_feat_dim if 'hand' in self.bhf_names else None
                            feat_dim_face = grid_feat_dim if 'face' in self.bhf_names else None
                        else:
                            if self.fuse_grid_align:
                                feat_dim_hand = bhf_att_feat_dim[
                                    'hand'] if 'hand' in self.bhf_names else None
                                feat_dim_face = bhf_att_feat_dim[
                                    'face'] if 'face' in self.bhf_names else None
                            else:
                                feat_dim_hand = bhf_ma_feat_dim[
                                    'hand'] if 'hand' in self.bhf_names else None
                                feat_dim_face = bhf_ma_feat_dim[
                                    'face'] if 'face' in self.bhf_names else None
                    else:
                        feat_dim_hand = ref_infeat_dim
                        feat_dim_face = ref_infeat_dim
                else:
                    ref_infeat_dim = global_feat_dim
                    feat_dim_hand = global_feat_dim
                    feat_dim_face = global_feat_dim

                self.regressor.append(
                    Regressor(
                        feat_dim=ref_infeat_dim,
                        smpl_mean_params=smpl_mean_params,
                        use_cam_feats=cfg.MODEL.PyMAF.USE_CAM_FEAT,
                        feat_dim_hand=feat_dim_hand,
                        feat_dim_face=feat_dim_face,
                        bhf_names=bhf_names,
                        smpl_models=self.smpl_family
                    )
                )

            # assign sub-regressor to each part
            for dec_name, dec_module in self.regressor[-1].named_children():
                if 'hand' in dec_name:
                    self.part_module_names['hand'].update(
                        {'regressor.{}.{}.'.format(len(self.regressor) - 1, dec_name): dec_module}
                    )
                elif 'face' in dec_name or 'head' in dec_name or 'exp' in dec_name:
                    self.part_module_names['face'].update(
                        {'regressor.{}.{}.'.format(len(self.regressor) - 1, dec_name): dec_module}
                    )
                elif 'res' in dec_name or 'vis' in dec_name:
                    self.part_module_names['link'].update(
                        {'regressor.{}.{}.'.format(len(self.regressor) - 1, dec_name): dec_module}
                    )
                elif 'body' in self.part_module_names:
                    self.part_module_names['body'].update(
                        {'regressor.{}.{}.'.format(len(self.regressor) - 1, dec_name): dec_module}
                    )

        # mesh-aligned feature extractor
        self.maf_extractor = nn.ModuleDict()
        for part in bhf_names:
            self.maf_extractor[part] = nn.ModuleList()
            filter_channels_default = cfg.MODEL.PyMAF.MLP_DIM if part == 'body' else cfg.MODEL.PyMAF.HF_MLP_DIM
            sfeat_dim = body_sfeat_dim if part == 'body' else hf_sfeat_dim
            for i in range(cfg.MODEL.PyMAF.N_ITER):
                for f_i, f_dim in enumerate(filter_channels_default):
                    if sfeat_dim[i] > f_dim:
                        filter_start = f_i
                        break
                filter_channels = [sfeat_dim[i]] + filter_channels_default[filter_start:]

                if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT and i >= self.att_starts:
                    self.maf_extractor[part].append(
                        MAF_Extractor(
                            filter_channels=filter_channels_default[att_feat_dim_idx:],
                            iwp_cam_mode=cfg.MODEL.USE_IWP_CAM
                        )
                    )
                else:
                    self.maf_extractor[part].append(
                        MAF_Extractor(
                            filter_channels=filter_channels, iwp_cam_mode=cfg.MODEL.USE_IWP_CAM
                        )
                    )
            self.part_module_names[part].update({f'maf_extractor.{part}': self.maf_extractor[part]})

        # check all modules have been added to part_module_names
        model_dict_all = dict.fromkeys(self.state_dict().keys())
        for key in self.part_module_names.keys():
            for name in list(model_dict_all.keys()):
                for k in self.part_module_names[key].keys():
                    if name.startswith(k):
                        del model_dict_all[name]
                # if name.startswith('regressor.') and '.smpl.' in name:
                #     del model_dict_all[name]
                # if name.startswith('regressor.') and '.mano.' in name:
                #     del model_dict_all[name]
                if name.startswith('regressor.') and '.init_' in name:
                    del model_dict_all[name]
                if name == 'grid_points':
                    del model_dict_all[name]
        assert (len(model_dict_all.keys()) == 0)

    def init_mesh(self, batch_size, J_regressor=None, rw_cam={}):
        """ initialize the mesh model with default poses and shapes
        """
        if self.init_mesh_output is None or self.batch_size != batch_size:
            self.init_mesh_output = self.regressor[0](
                torch.zeros(batch_size), J_regressor=J_regressor, rw_cam=rw_cam, init_mode=True
            )
            self.batch_size = batch_size
        return self.init_mesh_output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for m in ['body', 'hand', 'face']:
            if m in self.smpl_family:
                self.smpl_family[m].model.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        for m in ['body', 'hand', 'face']:
            if m in self.smpl_family:
                self.smpl_family[m].model.cuda(*args, **kwargs)
        return self

    def forward(self, batch={}, J_regressor=None, rw_cam={}):
        '''
        Args:
            batch: input dictionary, including 
                   images: 'img_{part}', for part in body, hand, and face if applicable
                   inversed affine transformation for the cropping of hand/face images: '{part}_theta_inv' for part in lhand, rhand, and face if applicable
            J_regressor: joint regression matrix
            rw_cam: real-world camera information, applied when cfg.MODEL.USE_IWP_CAM is False
        Returns:
            out_dict: the list containing the predicted parameters
            vis_feat_list: the list containing features for visualization
        '''

        # batch keys: ['img_body', 'orig_height', 'orig_width', 'person_id', 'img_lhand',
        # 'lhand_theta_inv', 'img_rhand', 'rhand_theta_inv', 'img_face', 'face_theta_inv']

        # extract spatial features or global features
        # run encoder for body
        if 'body' in self.bhf_names:
            img_body = batch['img_body']
            batch_size = img_body.shape[0]
            s_feat_body, g_feat = self.encoders['body'](batch['img_body'])
            if cfg.MODEL.PyMAF.MAF_ON:
                assert len(s_feat_body) == cfg.MODEL.PyMAF.N_ITER

        # run encoders for hand / face
        if 'hand' in self.bhf_names or 'face' in self.bhf_names:
            limb_feat_dict = {}
            limb_gfeat_dict = {}
            if 'face' in self.bhf_names:
                img_face = batch['img_face']
                batch_size = img_face.shape[0]
                limb_feat_dict['face'], limb_gfeat_dict['face'] = self.encoders['face'](img_face)

            if 'hand' in self.bhf_names:
                if 'lhand' in self.part_names:
                    img_rhand = batch['img_rhand']
                    batch_size = img_rhand.shape[0]
                    # flip left hand images
                    img_lhand = torch.flip(batch['img_lhand'], [3])
                    img_hands = torch.cat([img_rhand, img_lhand])
                    s_feat_hands, g_feat_hands = self.encoders['hand'](img_hands)
                    limb_feat_dict['rhand'] = [feat[:batch_size] for feat in s_feat_hands]
                    limb_feat_dict['lhand'] = [feat[batch_size:] for feat in s_feat_hands]
                    if g_feat_hands is not None:
                        limb_gfeat_dict['rhand'] = g_feat_hands[:batch_size]
                        limb_gfeat_dict['lhand'] = g_feat_hands[batch_size:]
                else:
                    img_rhand = batch['img_rhand']
                    batch_size = img_rhand.shape[0]
                    limb_feat_dict['rhand'], limb_gfeat_dict['rhand'] = self.encoders['hand'](
                        img_rhand
                    )

            if cfg.MODEL.PyMAF.MAF_ON:
                for k in limb_feat_dict.keys():
                    assert len(limb_feat_dict[k]) == cfg.MODEL.PyMAF.N_ITER

        out_dict = {}

        # grid-pattern points
        grid_points = torch.transpose(self.grid_points.expand(batch_size, -1, -1), 1, 2)

        # initial parameters
        mesh_output = self.init_mesh(batch_size, J_regressor, rw_cam)

        out_dict['mesh_out'] = [mesh_output]
        out_dict['dp_out'] = []

        # for visulization
        vis_feat_list = []

        # dense prediction during training
        if not cfg.MODEL.EVAL_MODE:
            if 'body' in self.bhf_names:
                if cfg.MODEL.PyMAF.AUX_SUPV_ON:
                    iuv_out_dict = self.dp_head(s_feat_body[-1])
                    out_dict['dp_out'].append(iuv_out_dict)
            elif self.hand_only_mode:
                if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                    out_dict['rhand_dpout'] = []
                    dphand_out_dict = self.dp_head_hf['hand'](limb_feat_dict['rhand'][-1])
                    out_dict['rhand_dpout'].append(dphand_out_dict)
            elif self.face_only_mode:
                if cfg.MODEL.PyMAF.HF_AUX_SUPV_ON:
                    out_dict['face_dpout'] = []
                    dpface_out_dict = self.dp_head_hf['face'](limb_feat_dict['face'][-1])
                    out_dict['face_dpout'].append(dpface_out_dict)

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            current_states = {}
            if 'body' in self.bhf_names:
                pred_cam = mesh_output['pred_cam'].detach()
                pred_shape = mesh_output['pred_shape'].detach()
                pred_pose = mesh_output['pred_pose'].detach()

                current_states['init_cam'] = pred_cam
                current_states['init_shape'] = pred_shape
                current_states['init_pose'] = pred_pose

                pred_smpl_verts = mesh_output['verts'].detach()

                if cfg.MODEL.PyMAF.MAF_ON:
                    s_feat_i = s_feat_body[rf_i]

            # re-project mesh on the image plane
            if self.hand_only_mode:
                pred_cam = mesh_output['pred_cam'].detach()
                pred_rhand_v = self.mano_sampler(mesh_output['verts_rh'])
                pred_rhand_proj = projection(
                    pred_rhand_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    }, iwp_mode=cfg.MODEL.USE_IWP_CAM
                )
                if cfg.MODEL.USE_IWP_CAM:
                    pred_rhand_proj = pred_rhand_proj / (224. / 2.)
                else:
                    pred_rhand_proj = j2d_processing(pred_rhand_proj, rw_cam['kps_transf'])
                proj_hf_center = {
                    'rhand':
                        mesh_output['pred_rhand_kp2d'][:, self.hf_root_idx['rhand']].unsqueeze(1)
                }
                proj_hf_pts = {
                    'rhand': torch.cat([proj_hf_center['rhand'], pred_rhand_proj], dim=1)
                }
            elif self.face_only_mode:
                pred_cam = mesh_output['pred_cam'].detach()
                pred_face_v = mesh_output['pred_face_kp3d']
                pred_face_proj = projection(
                    pred_face_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    }, iwp_mode=cfg.MODEL.USE_IWP_CAM
                )
                if cfg.MODEL.USE_IWP_CAM:
                    pred_face_proj = pred_face_proj / (224. / 2.)
                else:
                    pred_face_proj = j2d_processing(pred_face_proj, rw_cam['kps_transf'])
                proj_hf_center = {
                    'face': mesh_output['pred_face_kp2d'][:, self.hf_root_idx['face']].unsqueeze(1)
                }
                proj_hf_pts = {'face': torch.cat([proj_hf_center['face'], pred_face_proj], dim=1)}
            elif self.body_hand_mode:
                pred_lhand_v = self.mano_sampler(pred_smpl_verts[:, self.smpl2lhand])
                pred_rhand_v = self.mano_sampler(pred_smpl_verts[:, self.smpl2rhand])
                pred_hand_v = torch.cat([pred_lhand_v, pred_rhand_v], dim=1)
                pred_hand_proj = projection(
                    pred_hand_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    }, iwp_mode=cfg.MODEL.USE_IWP_CAM
                )
                if cfg.MODEL.USE_IWP_CAM:
                    pred_hand_proj = pred_hand_proj / (224. / 2.)
                else:
                    pred_hand_proj = j2d_processing(pred_hand_proj, rw_cam['kps_transf'])

                proj_hf_center = {
                    'lhand':
                        mesh_output['pred_lhand_kp2d'][:, self.hf_root_idx['lhand']].unsqueeze(1),
                    'rhand':
                        mesh_output['pred_rhand_kp2d'][:, self.hf_root_idx['rhand']].unsqueeze(1),
                }
                proj_hf_pts = {
                    'lhand':
                        torch.cat(
                            [proj_hf_center['lhand'], pred_hand_proj[:, :self.mano_ds_len]], dim=1
                        ),
                    'rhand':
                        torch.cat(
                            [proj_hf_center['rhand'], pred_hand_proj[:, self.mano_ds_len:]], dim=1
                        ),
                }
            elif self.full_body_mode:
                pred_lhand_v = self.mano_sampler(pred_smpl_verts[:, self.smpl2lhand])
                pred_rhand_v = self.mano_sampler(pred_smpl_verts[:, self.smpl2rhand])
                pred_hand_v = torch.cat([pred_lhand_v, pred_rhand_v], dim=1)
                pred_hand_proj = projection(
                    pred_hand_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    }, iwp_mode=cfg.MODEL.USE_IWP_CAM
                )
                if cfg.MODEL.USE_IWP_CAM:
                    pred_hand_proj = pred_hand_proj / (224. / 2.)
                else:
                    pred_hand_proj = j2d_processing(pred_hand_proj, rw_cam['kps_transf'])

                proj_hf_center = {
                    'lhand':
                        mesh_output['pred_lhand_kp2d'][:, self.hf_root_idx['lhand']].unsqueeze(1),
                    'rhand':
                        mesh_output['pred_rhand_kp2d'][:, self.hf_root_idx['rhand']].unsqueeze(1),
                    'face':
                        mesh_output['pred_face_kp2d'][:, self.hf_root_idx['face']].unsqueeze(1)
                }
                proj_hf_pts = {
                    'lhand':
                        torch.cat(
                            [proj_hf_center['lhand'], pred_hand_proj[:, :self.mano_ds_len]], dim=1
                        ),
                    'rhand':
                        torch.cat(
                            [proj_hf_center['rhand'], pred_hand_proj[:, self.mano_ds_len:]], dim=1
                        ),
                    'face':
                        torch.cat([proj_hf_center['face'], mesh_output['pred_face_kp2d']], dim=1)
                }

            # extract mesh-aligned features for the hand / face part
            if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                limb_rf_i = rf_i
                hand_face_feat = {}

                for hf_i, part_name in enumerate(self.part_names):
                    if 'hand' in part_name:
                        hf_key = 'hand'
                    elif 'face' in part_name:
                        hf_key = 'face'

                    if cfg.MODEL.PyMAF.MAF_ON:
                        if cfg.MODEL.PyMAF.HF_BACKBONE == 'res50':
                            limb_feat_i = limb_feat_dict[part_name][limb_rf_i]
                        else:
                            raise NotImplementedError

                        limb_reduce_dim = (not self.fuse_grid_align) or (rf_i < self.att_starts)

                        if limb_rf_i == 0 or cfg.MODEL.PyMAF.GRID_FEAT:
                            limb_ref_feat_ctd = self.maf_extractor[hf_key][limb_rf_i].sampling(
                                grid_points, im_feat=limb_feat_i, reduce_dim=limb_reduce_dim
                            )
                        else:
                            if self.hand_only_mode or self.face_only_mode:
                                proj_hf_pts_crop = proj_hf_pts[part_name][:, :, :2]

                                proj_hf_v_center = proj_hf_pts_crop[:, 0].unsqueeze(1)

                                if cfg.MODEL.PyMAF.HF_BOX_CENTER:
                                    part_box_ul = torch.min(proj_hf_pts_crop, dim=1)[0].unsqueeze(1)
                                    part_box_br = torch.max(proj_hf_pts_crop, dim=1)[0].unsqueeze(1)
                                    part_box_center = (part_box_ul + part_box_br) / 2.
                                    proj_hf_pts_crop_ctd = proj_hf_pts_crop[:, 1:] - part_box_center
                                else:
                                    proj_hf_pts_crop_ctd = proj_hf_pts_crop[:, 1:]

                            elif self.full_body_mode or self.body_hand_mode:
                                # convert projection points to the space of cropped hand/face images
                                theta_i_inv = batch[f'{part_name}_theta_inv']
                                proj_hf_pts_crop = torch.bmm(
                                    theta_i_inv,
                                    homo_vector(proj_hf_pts[part_name][:, :, :2]).permute(0, 2, 1)
                                ).permute(0, 2, 1)

                                if part_name == 'lhand':
                                    flip_x = torch.tensor([-1, 1])[None,
                                                                   None, :].to(proj_hf_pts_crop)
                                    proj_hf_pts_crop *= flip_x

                                if cfg.MODEL.PyMAF.HF_BOX_CENTER:
                                    # align projection points with the cropped image center
                                    part_box_ul = torch.min(proj_hf_pts_crop, dim=1)[0].unsqueeze(1)
                                    part_box_br = torch.max(proj_hf_pts_crop, dim=1)[0].unsqueeze(1)
                                    part_box_center = (part_box_ul + part_box_br) / 2.
                                    proj_hf_pts_crop_ctd = proj_hf_pts_crop[:, 1:] - part_box_center
                                else:
                                    proj_hf_pts_crop_ctd = proj_hf_pts_crop[:, 1:]

                                # 0 is the root point
                                proj_hf_v_center = proj_hf_pts_crop[:, 0].unsqueeze(1)

                            limb_ref_feat_ctd = self.maf_extractor[hf_key][limb_rf_i].sampling(
                                proj_hf_pts_crop_ctd.detach(),
                                im_feat=limb_feat_i,
                                reduce_dim=limb_reduce_dim
                            )

                        if self.fuse_grid_align and limb_rf_i >= self.att_starts:

                            limb_grid_feature_ctd = self.maf_extractor[hf_key][limb_rf_i].sampling(
                                grid_points, im_feat=limb_feat_i, reduce_dim=limb_reduce_dim
                            )
                            limb_grid_ref_feat_ctd = torch.cat(
                                [limb_grid_feature_ctd, limb_ref_feat_ctd], dim=-1
                            ).permute(0, 2, 1)

                            if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
                                att_ref_feat_ctd = self.align_attention[hf_key][
                                    limb_rf_i - self.att_starts](limb_grid_ref_feat_ctd)[0]
                            elif cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC:
                                att_ref_feat_ctd = limb_grid_ref_feat_ctd

                            att_ref_feat_ctd = self.maf_extractor[hf_key][limb_rf_i].reduce_dim(
                                att_ref_feat_ctd.permute(0, 2, 1)
                            ).view(batch_size, -1)
                            limb_ref_feat_ctd = self.att_feat_reduce[hf_key][
                                limb_rf_i - self.att_starts](att_ref_feat_ctd)

                        else:
                            # limb_ref_feat = limb_ref_feat.view(batch_size, -1)
                            limb_ref_feat_ctd = limb_ref_feat_ctd.view(batch_size, -1)
                        hand_face_feat[part_name] = limb_ref_feat_ctd
                    else:
                        hand_face_feat[part_name] = limb_gfeat_dict[part_name]

            # extract mesh-aligned features for the body part
            if 'body' in self.bhf_names:
                if cfg.MODEL.PyMAF.MAF_ON:
                    reduce_dim = (not self.fuse_grid_align) or (rf_i < self.att_starts)
                    if rf_i == 0 or cfg.MODEL.PyMAF.GRID_FEAT:
                        ref_feature = self.maf_extractor['body'][rf_i].sampling(
                            grid_points, im_feat=s_feat_i, reduce_dim=reduce_dim
                        )
                    else:
                        # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                        pred_smpl_verts_ds = self.mesh_sampler.downsample(
                            pred_smpl_verts
                        )    # [B, 431, 3]
                        ref_feature = self.maf_extractor['body'][rf_i](
                            pred_smpl_verts_ds,
                            im_feat=s_feat_i,
                            cam={
                                **rw_cam, 'cam_sxy': pred_cam
                            },
                            add_att=True,
                            reduce_dim=reduce_dim
                        )    # [B, 431 * n_feat]

                    if self.fuse_grid_align and rf_i >= self.att_starts:
                        if rf_i > 0 and not cfg.MODEL.PyMAF.GRID_FEAT:
                            grid_feature = self.maf_extractor['body'][rf_i].sampling(
                                grid_points, im_feat=s_feat_i, reduce_dim=reduce_dim
                            )
                            grid_ref_feat = torch.cat([grid_feature, ref_feature], dim=-1)
                        else:
                            grid_ref_feat = ref_feature
                        grid_ref_feat = grid_ref_feat.permute(0, 2, 1)

                        if cfg.MODEL.PyMAF.GRID_ALIGN.USE_ATT:
                            att_ref_feat = self.align_attention['body'][
                                rf_i - self.att_starts](grid_ref_feat)[0]
                        elif cfg.MODEL.PyMAF.GRID_ALIGN.USE_FC:
                            att_ref_feat = grid_ref_feat

                        att_ref_feat = self.maf_extractor['body'][rf_i].reduce_dim(
                            att_ref_feat.permute(0, 2, 1)
                        )
                        att_ref_feat = att_ref_feat.view(batch_size, -1)

                        ref_feature = self.att_feat_reduce['body'][rf_i -
                                                                   self.att_starts](att_ref_feat)
                    else:
                        ref_feature = ref_feature.view(batch_size, -1)
                else:
                    ref_feature = g_feat
            else:
                ref_feature = None

            if not self.smpl_mode:
                if self.hand_only_mode:
                    current_states['xc_rhand'] = hand_face_feat['rhand']
                elif self.face_only_mode:
                    current_states['xc_face'] = hand_face_feat['face']
                elif self.body_hand_mode:
                    current_states['xc_lhand'] = hand_face_feat['lhand']
                    current_states['xc_rhand'] = hand_face_feat['rhand']
                elif self.full_body_mode:
                    current_states['xc_lhand'] = hand_face_feat['lhand']
                    current_states['xc_rhand'] = hand_face_feat['rhand']
                    current_states['xc_face'] = hand_face_feat['face']

                if rf_i > 0:
                    for part in self.part_names:
                        current_states[f'init_{part}'] = mesh_output[f'pred_{part}'].detach()
                        if part == 'face':
                            current_states['init_exp'] = mesh_output['pred_exp'].detach()
                    if self.hand_only_mode:
                        current_states['init_shape_rh'] = mesh_output['pred_shape_rh'].detach()
                        current_states['init_orient_rh'] = mesh_output['pred_orient_rh'].detach()
                        current_states['init_cam_rh'] = mesh_output['pred_cam_rh'].detach()
                    elif self.face_only_mode:
                        current_states['init_shape_fa'] = mesh_output['pred_shape_fa'].detach()
                        current_states['init_orient_fa'] = mesh_output['pred_orient_fa'].detach()
                        current_states['init_cam_fa'] = mesh_output['pred_cam_fa'].detach()
                    elif self.full_body_mode or self.body_hand_mode:
                        if cfg.MODEL.PyMAF.OPT_WRIST:
                            current_states['init_shape_lh'] = mesh_output['pred_shape_lh'].detach()
                            current_states['init_orient_lh'] = mesh_output['pred_orient_lh'].detach(
                            )
                            current_states['init_cam_lh'] = mesh_output['pred_cam_lh'].detach()

                            current_states['init_shape_rh'] = mesh_output['pred_shape_rh'].detach()
                            current_states['init_orient_rh'] = mesh_output['pred_orient_rh'].detach(
                            )
                            current_states['init_cam_rh'] = mesh_output['pred_cam_rh'].detach()

            # update mesh parameters
            mesh_output = self.regressor[rf_i](
                ref_feature,
                n_iter=1,
                J_regressor=J_regressor,
                rw_cam=rw_cam,
                global_iter=rf_i,
                **current_states
            )

            out_dict['mesh_out'].append(mesh_output)

        return out_dict, vis_feat_list


def pymaf_net(smpl_mean_params, pretrained=True, device=torch.device('cuda')):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params, pretrained, device)
    return model
