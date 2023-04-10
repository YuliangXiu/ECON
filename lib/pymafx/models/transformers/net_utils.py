import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """
    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x, pt_coord=None):
        b, c, h, w = x.shape
        if pt_coord is not None:
            z_embed = pt_coord[:, :, 2].unsqueeze(-1) + 1.
            y_embed = pt_coord[:, :, 1].unsqueeze(-1) + 1.
            x_embed = pt_coord[:, :, 0].unsqueeze(-1) + 1.
        else:
            not_mask = torch.ones(1, h, w, device=x.device)
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            z_embed = torch.ones_like(x_embed)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (torch.max(z_embed) + eps) * self.scale
            y_embed = y_embed / (torch.max(y_embed) + eps) * self.scale
            x_embed = x_embed / (torch.max(x_embed) + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                            dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                            dim=4).flatten(3)
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()),
                            dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=3).permute(0, 3, 1, 2)
        # if pt_coord is None:
        pos = pos.repeat(b, 1, 1, 1)
        return pos


def softmax_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)    # b x n x hw x d
    k = k.flatten(-2)    # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)

    print('softmax', q.shape, k.shape, v.shape)

    N = k.shape[-1]    # ?????? maybe change to k.shape[-2]????
    attn = torch.matmul(q / N**0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn


def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)    # b x n x hw x d
    k = k.flatten(-2)    # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)

    N = k.shape[-1]
    attn = None
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn


def long_range_attention(q, k, v, P_h, P_w):    # fixed patch size
    B, N, C, qH, qW = q.size()
    _, _, _, kH, kW = k.size()

    qQ_h, qQ_w = qH // P_h, qW // P_w
    kQ_h, kQ_w = kH // P_h, kW // P_w

    q = q.reshape(B, N, C, qQ_h, P_h, qQ_w, P_w)
    k = k.reshape(B, N, C, kQ_h, P_h, kQ_w, P_w)
    v = v.reshape(B, N, -1, kQ_h, P_h, kQ_w, P_w)

    q = q.permute(0, 1, 4, 6, 2, 3, 5)    # [b, n, Ph, Pw, d, Qh, Qw]
    k = k.permute(0, 1, 4, 6, 2, 3, 5)
    v = v.permute(0, 1, 4, 6, 2, 3, 5)

    output, attn = softmax_attention(q, k, v)    # attn: [b, n, Ph, Pw, qQh*qQw, kQ_h*kQ_w]
    output = output.permute(0, 1, 4, 5, 2, 6, 3)
    output = output.reshape(B, N, -1, qH, qW)
    return output, attn


def short_range_attention(q, k, v, Q_h, Q_w):    # fixed patch number
    B, N, C, qH, qW = q.size()
    _, _, _, kH, kW = k.size()

    qP_h, qP_w = qH // Q_h, qW // Q_w
    kP_h, kP_w = kH // Q_h, kW // Q_w

    q = q.reshape(B, N, C, Q_h, qP_h, Q_w, qP_w)
    k = k.reshape(B, N, C, Q_h, kP_h, Q_w, kP_w)
    v = v.reshape(B, N, -1, Q_h, kP_h, Q_w, kP_w)

    q = q.permute(0, 1, 3, 5, 2, 4, 6)    # [b, n, Qh, Qw, d, Ph, Pw]
    k = k.permute(0, 1, 3, 5, 2, 4, 6)
    v = v.permute(0, 1, 3, 5, 2, 4, 6)

    output, attn = softmax_attention(q, k, v)    # attn: [b, n, Qh, Qw, qPh*qPw, kPh*kPw]
    output = output.permute(0, 1, 4, 2, 5, 3, 6)
    output = output.reshape(B, N, -1, qH, qW)
    return output, attn


def space_to_depth(x, block_size):
    x_shape = x.shape
    c, h, w = x_shape[-3:]
    if len(x.shape) >= 5:
        x = x.view(-1, c, h, w)
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(*x_shape[0:-3], c * block_size**2, h // block_size, w // block_size)


def depth_to_space(x, block_size):
    x_shape = x.shape
    c, h, w = x_shape[-3:]
    x = x.view(-1, c, h, w)
    y = torch.nn.functional.pixel_shuffle(x, block_size)
    return y.view(*x_shape[0:-3], -1, h * block_size, w * block_size)


def patch_attention(q, k, v, P):
    # q: [b, nhead, c, h, w]
    q_patch = space_to_depth(q, P)    # [b, nhead, cP^2, h/P, w/P]
    k_patch = space_to_depth(k, P)
    v_patch = space_to_depth(v, P)

    # output: [b, nhead, cP^2, h/P, w/P]
    # attn: [b, nhead, h/P*w/P, h/P*w/P]
    output, attn = softmax_attention(q_patch, k_patch, v_patch)
    output = depth_to_space(output, P)    # output: [b, nhead, c, h, w]
    return output, attn
