# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.net.BasePIFuNet import BasePIFuNet
from lib.net.FBNet import GANLoss, IDMRFLoss, VGGLoss, define_D, define_G
from lib.net.net_util import init_net


class NormalNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg):

        super(NormalNet, self).__init__()

        self.opt = cfg.net

        self.F_losses = [item[0] for item in self.opt.front_losses]
        self.B_losses = [item[0] for item in self.opt.back_losses]
        self.F_losses_ratio = [item[1] for item in self.opt.front_losses]
        self.B_losses_ratio = [item[1] for item in self.opt.back_losses]
        self.ALL_losses = self.F_losses + self.B_losses

        if self.training:
            if 'vgg' in self.ALL_losses:
                self.vgg_loss = VGGLoss()
            if ('gan' in self.ALL_losses) or ('gan_feat' in self.ALL_losses):
                self.gan_loss = GANLoss(use_lsgan=True)
            if 'mrf' in self.ALL_losses:
                self.mrf_loss = IDMRFLoss()
            if 'l1' in self.ALL_losses:
                self.l1_loss = nn.SmoothL1Loss()

        self.in_nmlF = [
            item[0] for item in self.opt.in_nml if "_F" in item[0] or item[0] == "image"
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml if "_B" in item[0] or item[0] == "image"
        ]
        self.in_nmlF_dim = sum([
            item[1] for item in self.opt.in_nml if "_F" in item[0] or item[0] == "image"
        ])
        self.in_nmlB_dim = sum([
            item[1] for item in self.opt.in_nml if "_B" in item[0] or item[0] == "image"
        ])

        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

        if ('gan' in self.ALL_losses):
            self.netD = define_D(3, 64, 3, 'instance', False, 2, 'gan_feat' in self.ALL_losses)

        init_net(self)

    def forward(self, in_tensor):

        inF_list = []
        inB_list = []

        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])
        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])

        nmlF = self.netF(torch.cat(inF_list, dim=1))
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        # ||normal|| == 1
        nmlF_normalized = nmlF / torch.norm(nmlF, dim=1, keepdim=True)
        nmlB_normalized = nmlB / torch.norm(nmlB, dim=1, keepdim=True)

        # output: float_arr [-1,1] with [B, C, H, W]
        mask = ((in_tensor["image"].abs().sum(dim=1, keepdim=True) != 0.0).detach().float())

        return nmlF_normalized * mask, nmlB_normalized * mask

    def get_norm_error(self, prd_F, prd_B, tgt):
        """calculate normal loss

        Args:
            pred (torch.tensor): [B, 6, 512, 512]
            tagt (torch.tensor): [B, 6, 512, 512]
        """

        tgt_F, tgt_B = tgt["normal_F"], tgt["normal_B"]

        # netF, netB, netD
        total_loss = {"netF": 0.0, "netB": 0.0}

        if 'l1' in self.F_losses:
            l1_F_loss = self.l1_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
            total_loss["l1_F"] = self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
        if 'l1' in self.B_losses:
            l1_B_loss = self.l1_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss
            total_loss["l1_B"] = self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss

        if 'vgg' in self.F_losses:
            vgg_F_loss = self.vgg_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
            total_loss["vgg_F"] = self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
        if 'vgg' in self.B_losses:
            vgg_B_loss = self.vgg_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss
            total_loss["vgg_B"] = self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss

        scale_factor = 0.5
        if 'mrf' in self.F_losses:
            mrf_F_loss = self.mrf_loss(
                F.interpolate(prd_F, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_F, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
            total_loss["mrf_F"] = self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
        if 'mrf' in self.B_losses:
            mrf_B_loss = self.mrf_loss(
                F.interpolate(prd_B, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_B, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss
            total_loss["mrf_B"] = self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss

        if 'gan' in self.ALL_losses:

            total_loss["netD"] = 0.0

            pred_fake = self.netD.forward(prd_B)
            pred_real = self.netD.forward(tgt_B)
            loss_D_fake = self.gan_loss(pred_fake, False)
            loss_D_real = self.gan_loss(pred_real, True)
            loss_G_fake = self.gan_loss(pred_fake, True)

            total_loss["netD"] += 0.5 * (loss_D_fake + loss_D_real
                                        ) * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_fake"] = loss_D_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_real"] = loss_D_real * self.B_losses_ratio[self.B_losses.index('gan')]

            total_loss["netB"] += loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["G_fake"] = loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]

            if 'gan_feat' in self.ALL_losses:
                loss_G_GAN_Feat = 0
                for i in range(2):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += self.l1_loss(pred_fake[i][j], pred_real[i][j].detach())
                total_loss["netB"] += loss_G_GAN_Feat * self.B_losses_ratio[
                    self.B_losses.index('gan_feat')]
                total_loss["G_GAN_Feat"] = loss_G_GAN_Feat * self.B_losses_ratio[
                    self.B_losses.index('gan_feat')]

        return total_loss
