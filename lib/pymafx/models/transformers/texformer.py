import torch.nn as nn

from .net_utils import (
    PosEnSine,
    double_conv,
    double_conv_down,
    double_conv_up,
    single_conv,
)
from .transformer_basics import OurMultiheadAttention


class TransformerDecoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, attn_type='softmax', P=None):
        super(TransformerDecoderUnit, self).__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = OurMultiheadAttention(feat_dim, n_head)    # cross-attention

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(self.feat_dim)

    def forward(self, q, k, v):
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0

        # cross-multi-head attention
        out = self.attn(
            q=q + q_pos_embed, k=k + k_pos_embed, v=v, attn_type=self.attn_type, P=self.P
        )[0]

        # feed forward
        out2 = self.linear2(self.activation(self.linear1(out)))
        out = out + out2
        out = self.norm(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = double_conv_down(feat_ch, feat_ch)
        self.conv2 = double_conv_down(feat_ch, feat_ch)
        self.conv3 = double_conv(feat_ch, feat_ch)
        self.conv4 = double_conv_up(feat_ch, feat_ch)
        self.conv5 = double_conv_up(feat_ch, feat_ch)
        self.conv6 = double_conv(feat_ch, out_ch)

    def forward(self, x):
        feat0 = self.conv_in(x)    # H
        feat1 = self.conv1(feat0)    # H/2
        feat2 = self.conv2(feat1)    # H/4
        feat3 = self.conv3(feat2)    # H/4
        feat3 = feat3 + feat2    # H/4
        feat4 = self.conv4(feat3)    # H/2
        feat4 = feat4 + feat1    # H/2
        feat5 = self.conv5(feat4)    # H
        feat5 = feat5 + feat0    # H
        feat6 = self.conv6(feat5)

        return feat0, feat1, feat2, feat3, feat4, feat6


class Texformer(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.feat_dim = opts.feat_dim
        src_ch = opts.src_ch
        tgt_ch = opts.tgt_ch
        out_ch = opts.out_ch
        self.mask_fusion = opts.mask_fusion

        if not self.mask_fusion:
            v_ch = out_ch
        else:
            v_ch = 2 + 3

        self.unet_q = Unet(tgt_ch, self.feat_dim, self.feat_dim)
        self.unet_k = Unet(src_ch, self.feat_dim, self.feat_dim)
        self.unet_v = Unet(v_ch, self.feat_dim, self.feat_dim)

        self.trans_dec = nn.ModuleList([
            None, None, None,
            TransformerDecoderUnit(self.feat_dim, opts.nhead, True, 'softmax'),
            TransformerDecoderUnit(self.feat_dim, opts.nhead, True, 'dotproduct'),
            TransformerDecoderUnit(self.feat_dim, opts.nhead, True, 'dotproduct')
        ])

        self.conv0 = double_conv(self.feat_dim, self.feat_dim)
        self.conv1 = double_conv_down(self.feat_dim, self.feat_dim)
        self.conv2 = double_conv_down(self.feat_dim, self.feat_dim)
        self.conv3 = double_conv(self.feat_dim, self.feat_dim)
        self.conv4 = double_conv_up(self.feat_dim, self.feat_dim)
        self.conv5 = double_conv_up(self.feat_dim, self.feat_dim)

        if not self.mask_fusion:
            self.conv6 = nn.Sequential(
                single_conv(self.feat_dim, self.feat_dim),
                nn.Conv2d(self.feat_dim, out_ch, 3, 1, 1)
            )
        else:
            self.conv6 = nn.Sequential(
                single_conv(self.feat_dim, self.feat_dim),
                nn.Conv2d(self.feat_dim, 2 + 3 + 1, 3, 1, 1)
            )    # mask*flow-sampling + (1-mask)*rgb
            self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

    def forward(self, q, k, v):
        print('qkv', q.shape, k.shape, v.shape)
        q_feat = self.unet_q(q)
        k_feat = self.unet_k(k)
        v_feat = self.unet_v(v)

        print('q_feat', len(q_feat))
        outputs = []
        for i in range(3, len(q_feat)):
            print(i, q_feat[i].shape, k_feat[i].shape, v_feat[i].shape)
            outputs.append(self.trans_dec[i](q_feat[i], k_feat[i], v_feat[i]))
            print('outputs', outputs[-1].shape)

        f0 = self.conv0(outputs[2])    # H
        f1 = self.conv1(f0)    # H/2
        f1 = f1 + outputs[1]
        f2 = self.conv2(f1)    # H/4
        f2 = f2 + outputs[0]
        f3 = self.conv3(f2)    # H/4
        f3 = f3 + outputs[0] + f2
        f4 = self.conv4(f3)    # H/2
        f4 = f4 + outputs[1] + f1
        f5 = self.conv5(f4)    # H
        f5 = f5 + outputs[2] + f0
        if not self.mask_fusion:
            out = self.tanh(self.conv6(f5))
        else:
            out_ = self.conv6(f5)
            out = [self.tanh(out_[:, :2]), self.tanh(out_[:, 2:5]), self.sigmoid(out_[:, 5:])]
        return out
