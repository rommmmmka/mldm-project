import torch


class ChannelShuffle(torch.nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, length = x.size()
        x = x.view(batch_size, self.groups, channels // self.groups, length)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, channels, length)
        return x


# class AdaBN(torch.nn.Module):
#     def __init__(self, num_features):
#         super(AdaBN, self).__init__()
#         self.gamma = torch.nn.Parameter(torch.ones(num_features))
#         self.beta = torch.nn.Parameter(torch.zeros(num_features))
#
#     def forward(self, x, adaptive_params=None):
#         if adaptive_params is None:
#             mean = x.mean(dim=(0, 2), keepdim=True)
#             var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
#         else:
#             mean, var = adaptive_params
#
#         x_norm = (x - mean) / (var.sqrt() + 1e-5)
#         return self.gamma[None, :, None] * x_norm + self.beta[None, :, None]


class GlobalAveragePooling(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


class ResidualBlock1(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock1, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 64, kernel_size=16, padding=8, groups=8)
        # self.adabn1 = AdaBN(64)
        self.batchnorm1  = torch.nn.BatchNorm1d(64)
        self.shuffle1 = ChannelShuffle(8)
        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=16, padding=7, groups=8)
        # self.adabn2 = AdaBN(64)
        self.batchnorm2  = torch.nn.BatchNorm1d(64)
        self.shuffle2 = ChannelShuffle(8)

    def forward(self, x, adaptive_params):
        out = self.conv1(x)
        # out = self.adabn1(out, adaptive_params)
        out = self.batchnorm1(out)
        out = torch.relu(out)
        out = self.shuffle1(out)

        out = self.conv2(out)
        # out = self.adabn2(out, adaptive_params)
        out = self.batchnorm2(out)
        out = torch.relu(out)
        out = self.shuffle2(out)

        out += x
        return out


class ResidualBlock2(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock2, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 128, kernel_size=16, padding=8, stride=2, groups=16)
        # self.adabn1 = AdaBN(128)
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.shuffle1 = ChannelShuffle(16)
        self.conv2 = torch.nn.Conv1d(128, 128, kernel_size=16, padding=7, groups=16)
        # self.adabn2 = AdaBN(128)
        self.batchnorm2 = torch.nn.BatchNorm1d(128)
        self.shuffle2 = ChannelShuffle(16)

        self.match_dimensions = torch.nn.Conv1d(64, 128, kernel_size=1, stride=2)

    def forward(self, x, adaptive_params):
        out = self.conv1(x)
        # out = self.adabn1(out, adaptive_params)
        out = self.batchnorm1(out)
        out = torch.relu(out)
        out = self.shuffle1(out)

        out = self.conv2(out)
        # out = self.adabn2(out, adaptive_params)
        out = self.batchnorm2(out)
        out = torch.relu(out)
        out = self.shuffle2(out)

        x = self.match_dimensions(x)

        out += x
        return out


class LightSleepNet(torch.nn.Module):
    def __init__(self):
        super(LightSleepNet, self).__init__()
        self.conv = torch.nn.Conv1d(1, 64, kernel_size=16, padding=7, stride=2)
        # self.adabn = AdaBN(64)
        self.batchnorm  = torch.nn.BatchNorm1d(64)
        self.residual1 = ResidualBlock1()
        self.residual2 = ResidualBlock2()
        self.dropout = torch.nn.Dropout(0.5)
        self.pooling = GlobalAveragePooling()
        self.linear = torch.nn.Linear(128, 5)

    def forward(self, x, adaptive_params=None):
        x = self.conv(x)
        # x = self.adabn(x, adaptive_params)
        x = self.batchnorm(x)
        x = torch.relu(x)

        x = self.residual1(x, adaptive_params)
        x = self.residual2(x, adaptive_params)

        x = self.dropout(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
