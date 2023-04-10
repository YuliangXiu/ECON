""" The code is based on https://github.com/apple/ml-gsn/ with adaption. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from lib.net.Discriminator import StyleDiscriminator


def hinge_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        d_loss_real = torch.mean(F.relu(1.0 - real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = -torch.mean(fake_pred)
    return d_loss


def logistic_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.softplus(fake_pred))
        d_loss_real = torch.mean(F.softplus(-real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = torch.mean(F.softplus(-fake_pred))
    return d_loss


def r1_loss(real_pred, real_img):
    (grad_real, ) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class GANLoss(nn.Module):
    def __init__(
        self,
        opt,
        disc_loss='logistic',
    ):
        super().__init__()
        self.opt = opt.gan

        input_dim = 3
        self.discriminator = StyleDiscriminator(input_dim, self.opt.img_res)

        if disc_loss == 'hinge':
            self.disc_loss = hinge_loss
        elif disc_loss == 'logistic':
            self.disc_loss = logistic_loss

    def forward(self, input):

        disc_in_real = input['norm_real']
        disc_in_fake = input['norm_fake']

        logits_real = self.discriminator(disc_in_real)
        logits_fake = self.discriminator(disc_in_fake)

        disc_loss = self.disc_loss(fake_pred=logits_fake, real_pred=logits_real, mode='d')

        log = {
            "disc_loss": disc_loss.detach(),
            "logits_real": logits_real.mean().detach(),
            "logits_fake": logits_fake.mean().detach(),
        }

        return disc_loss * self.opt.lambda_gan, log
