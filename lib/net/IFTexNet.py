import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net.geometry import orthogonal


class IFTexNet(nn.Module):
    def __init__(self, hidden_dim=256):
        super(IFTexNet, self).__init__()

        self.conv_in = nn.Conv3d(
            4, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1,
                                padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(
            32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1,
                                padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(
            64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (4 + 16 + 32 + 64 + 128 + 128 + 128) + 39
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 3, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.InstanceNorm3d(16)
        self.conv0_1_bn = nn.InstanceNorm3d(32)
        self.conv1_1_bn = nn.InstanceNorm3d(64)
        self.conv2_1_bn = nn.InstanceNorm3d(128)
        self.conv3_1_bn = nn.InstanceNorm3d(128)
        self.conv4_1_bn = nn.InstanceNorm3d(128)

        self.freq_encoder = FreqEncoder()

    def forward(self, batch):
        
        x = batch["voxels_color"]
        p = orthogonal(batch["samples_color"].permute(0, 2, 1),
                       batch["calib"]).permute(0, 2, 1)  #[2, 60000, 3]
        
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        feature_0 = F.grid_sample(x, p, padding_mode='border', align_corners = True)
        # print(feature_0.shape)
        # print(feature_0[:,:,:,0,0])

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border', align_corners = True)
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border', align_corners = True)
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border', align_corners = True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border', align_corners = True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border', align_corners = True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border', align_corners = True)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        # (B, featues_per_sample, samples_num)
        features = torch.reshape(
            features, (shape[0], shape[1] * shape[3], shape[4]))
        # (B, featue_size, samples_num) samples_num 0->0,...,N->N

        p_features = p_features.permute(0, 2, 1)
        p_features = self.freq_encoder(p_features)
        p_features = p_features.permute(0, 2, 1)

        features = torch.cat((features, p_features), dim=1)
        # print('features: ', features[:,:,:3])

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out = self.fc_out(net)
        # out = net.squeeze(1)

        return out


class FreqEncoder(nn.Module):
    def __init__(self, input_dim=3, max_freq_log2=5, N_freqs=6,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out
