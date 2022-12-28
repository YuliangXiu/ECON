import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1), nn.BatchNorm2d(1), nn.ReLU()
        )

        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, C, H, W)

        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map


class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.Tensor(B, self.S, C).to(x)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)    # [B, C]
            Z[:, i, :] = Ai
        return Z


class TokenFuser(nn.Module):
    def __init__(self, H, W, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S

    def forward(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape

        Y = self.projection(y.reshape(B, C, S)).reshape(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).reshape(B, H * W, S)    # [B, HW, S]
        BwY = torch.matmul(Bw, Y)

        _, xj = self.spatial_attn(x)
        xj = xj.reshape(B, H * W, C)

        out = (BwY + xj).reshape(B, H, W, C)

        return out
