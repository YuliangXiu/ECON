import torch.nn as nn
from .net_utils import PosEnSine, softmax_attention, dotproduct_attention, long_range_attention, \
                                   short_range_attention, patch_attention


class OurMultiheadAttention(nn.Module):
    def __init__(self, q_feat_dim, k_feat_dim, out_feat_dim, n_head, d_k=None, d_v=None):
        super(OurMultiheadAttention, self).__init__()
        if d_k is None:
            d_k = out_feat_dim // n_head
        if d_v is None:
            d_v = out_feat_dim // n_head

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(q_feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(k_feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(out_feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, out_feat_dim, 1, bias=False)

    def forward(self, q, k, v, attn_type='softmax', **kwargs):
        # input: b x d x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection: b x (nhead*dk) x h x w
        # Separate different heads: b x nhead x dk x h x w
        q = self.w_qs(q).view(q.shape[0], n_head, d_k, q.shape[2], q.shape[3])
        k = self.w_ks(k).view(k.shape[0], n_head, d_k, k.shape[2], k.shape[3])
        v = self.w_vs(v).view(v.shape[0], n_head, d_v, v.shape[2], v.shape[3])

        # -------------- Attention -----------------
        if attn_type == 'softmax':
            q, attn = softmax_attention(q, k, v)    # b x n x dk x h x w --> b x n x dv x h x w
        elif attn_type == 'dotproduct':
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == 'patch':
            q, attn = patch_attention(q, k, v, P=kwargs['P'])
        elif attn_type == 'sparse_long':
            q, attn = long_range_attention(q, k, v, P_h=kwargs['ah'], P_w=kwargs['aw'])
        elif attn_type == 'sparse_short':
            q, attn = short_range_attention(q, k, v, Q_h=kwargs['ah'], Q_w=kwargs['aw'])
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')
        # ------------ end Attention ---------------

        # Concatenate all the heads together: b x (n*dv) x h x w
        q = q.reshape(q.shape[0], -1, q.shape[3], q.shape[4])
        q = self.fc(q)    # b x d x h x w

        return q, attn


class TransformerEncoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, attn_type='softmax', P=None):
        super(TransformerEncoderUnit, self).__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = OurMultiheadAttention(feat_dim, n_head)

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.feat_dim)
        self.norm2 = nn.BatchNorm2d(self.feat_dim)

    def forward(self, src):
        if self.pos_en_flag:
            pos_embed = self.pos_en(src)
        else:
            pos_embed = 0

        # multi-head attention
        src2 = self.attn(
            q=src + pos_embed, k=src + pos_embed, v=src, attn_type=self.attn_type, P=self.P
        )[0]
        src = src + src2
        src = self.norm1(src)

        # feed forward
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)

        return src


class TransformerEncoderUnitSparse(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, ahw=None):
        super(TransformerEncoderUnitSparse, self).__init__()
        self.feat_dim = feat_dim
        self.pos_en_flag = pos_en_flag
        self.ahw = ahw    # [Ph, Pw, Qh, Qw]

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn1 = OurMultiheadAttention(feat_dim, n_head)    # long range
        self.attn2 = OurMultiheadAttention(feat_dim, n_head)    # short range

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.feat_dim)
        self.norm2 = nn.BatchNorm2d(self.feat_dim)

    def forward(self, src):
        if self.pos_en_flag:
            pos_embed = self.pos_en(src)
        else:
            pos_embed = 0

        # multi-head long-range attention
        src2 = self.attn1(
            q=src + pos_embed,
            k=src + pos_embed,
            v=src,
            attn_type='sparse_long',
            ah=self.ahw[0],
            aw=self.ahw[1]
        )[0]
        src = src + src2    # ? this might be ok to remove

        # multi-head short-range attention
        src2 = self.attn2(
            q=src + pos_embed,
            k=src + pos_embed,
            v=src,
            attn_type='sparse_short',
            ah=self.ahw[2],
            aw=self.ahw[3]
        )[0]
        src = src + src2
        src = self.norm1(src)

        # feed forward
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)

        return src


class TransformerDecoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, attn_type='softmax', P=None):
        super(TransformerDecoderUnit, self).__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn1 = OurMultiheadAttention(feat_dim, n_head)    # self-attention
        self.attn2 = OurMultiheadAttention(feat_dim, n_head)    # cross-attention

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.feat_dim)
        self.norm2 = nn.BatchNorm2d(self.feat_dim)
        self.norm3 = nn.BatchNorm2d(self.feat_dim)

    def forward(self, tgt, src):
        if self.pos_en_flag:
            src_pos_embed = self.pos_en(src)
            tgt_pos_embed = self.pos_en(tgt)
        else:
            src_pos_embed = 0
            tgt_pos_embed = 0

        # self-multi-head attention
        tgt2 = self.attn1(
            q=tgt + tgt_pos_embed, k=tgt + tgt_pos_embed, v=tgt, attn_type=self.attn_type, P=self.P
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # cross-multi-head attention
        tgt2 = self.attn2(
            q=tgt + tgt_pos_embed, k=src + src_pos_embed, v=src, attn_type=self.attn_type, P=self.P
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderUnitSparse(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, ahw=None):
        super(TransformerDecoderUnitSparse, self).__init__()
        self.feat_dim = feat_dim
        self.ahw = ahw    # [Ph_tgt, Pw_tgt, Qh_tgt, Qw_tgt, Ph_src, Pw_src, Qh_tgt, Qw_tgt]
        self.pos_en_flag = pos_en_flag

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn1_1 = OurMultiheadAttention(feat_dim, n_head)    # self-attention: long
        self.attn1_2 = OurMultiheadAttention(feat_dim, n_head)    # self-attention: short

        self.attn2_1 = OurMultiheadAttention(
            feat_dim, n_head
        )    # cross-attention: self-attention-long + cross-attention-short
        self.attn2_2 = OurMultiheadAttention(feat_dim, n_head)

        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(self.feat_dim)
        self.norm2 = nn.BatchNorm2d(self.feat_dim)
        self.norm3 = nn.BatchNorm2d(self.feat_dim)

    def forward(self, tgt, src):
        if self.pos_en_flag:
            src_pos_embed = self.pos_en(src)
            tgt_pos_embed = self.pos_en(tgt)
        else:
            src_pos_embed = 0
            tgt_pos_embed = 0

        # self-multi-head attention: sparse long
        tgt2 = self.attn1_1(
            q=tgt + tgt_pos_embed,
            k=tgt + tgt_pos_embed,
            v=tgt,
            attn_type='sparse_long',
            ah=self.ahw[0],
            aw=self.ahw[1]
        )[0]
        tgt = tgt + tgt2
        # self-multi-head attention: sparse short
        tgt2 = self.attn1_2(
            q=tgt + tgt_pos_embed,
            k=tgt + tgt_pos_embed,
            v=tgt,
            attn_type='sparse_short',
            ah=self.ahw[2],
            aw=self.ahw[3]
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # self-multi-head attention: sparse long
        src2 = self.attn2_1(
            q=src + src_pos_embed,
            k=src + src_pos_embed,
            v=src,
            attn_type='sparse_long',
            ah=self.ahw[4],
            aw=self.ahw[5]
        )[0]
        src = src + src2
        # cross-multi-head attention: sparse short
        tgt2 = self.attn2_2(
            q=tgt + tgt_pos_embed,
            k=src + src_pos_embed,
            v=src,
            attn_type='sparse_short',
            ah=self.ahw[6],
            aw=self.ahw[7]
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt
