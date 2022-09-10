# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import pytorch_lightning as pl
import numpy as np


class SpatialEncoder(pl.LightningModule):

    def __init__(self,
                 sp_level=1,
                 sp_type="rel_z_decay",
                 scale=1.0,
                 n_kpt=24,
                 sigma=0.2):

        super().__init__()

        self.sp_type = sp_type
        self.sp_level = sp_level
        self.n_kpt = n_kpt
        self.scale = scale
        self.sigma = sigma

    @staticmethod
    def position_embedding(x, nlevels, scale=1.0):
        """
        args:
            x: (B, N, C)
        return:
            (B, N, C * n_levels * 2)
        """
        if nlevels <= 0:
            return x
        vec = SpatialEncoder.pe_vector(nlevels, x.device, scale)

        B, N, _ = x.shape
        y = x[:, :, None, :] * vec[None, None, :, None]
        z = torch.cat((torch.sin(y), torch.cos(y)), axis=-1).view(B, N, -1)

        return torch.cat([x, z], -1)

    @staticmethod
    def pe_vector(nlevels, device, scale=1.0):
        v, val = [], 1
        for _ in range(nlevels):
            v.append(scale * np.pi * val)
            val *= 2
        return torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)

    def get_dim(self):
        if self.sp_type in ["z", "rel_z", "rel_z_decay"]:
            if "rel" in self.sp_type:
                return (1 + 2 * self.sp_level) * self.n_kpt
            else:
                return 1 + 2 * self.sp_level
        elif "xyz" in self.sp_type:
            if "rel" in self.sp_type:
                return (1 + 2 * self.sp_level) * 3 * self.n_kpt
            else:
                return (1 + 2 * self.sp_level) * 3

        return 0

    def forward(self, cxyz, kptxyz):

        B, N = cxyz.shape[:2]
        K = kptxyz.shape[1]

        dz = cxyz[:, :, None, 2:3] - kptxyz[:, None, :, 2:3]
        dxyz = cxyz[:, :, None] - kptxyz[:, None, :]
        
        # (B, N, K)
        weight = torch.exp(-(dxyz**2).sum(-1) / (2.0 * (self.sigma**2)))

        # position embedding ( B, N, K * (2*n_levels+1) )
        out = self.position_embedding(dz.view(B, N, K), self.sp_level)
        
        # BV,N,K,(2*n_levels+1) * B,N,K,1 = B,N,K*(2*n_levels+1) -> BV,K*(2*n_levels+1),N
        out = (out.view(B, N, -1, K) * weight[:, :, None]).view(B, N, -1).permute(0,2,1) 

        return out


if __name__ == "__main__":
    pts = torch.randn(2, 10000, 3).to("cuda")
    kpts = torch.randn(2, 24, 3).to("cuda")

    sp_encoder = SpatialEncoder(sp_level=3,
                                sp_type="rel_z_decay",
                                scale=1.0,
                                n_kpt=24,
                                sigma=0.1).to("cuda")
    out = sp_encoder(pts, kpts)
    print(out.shape)
