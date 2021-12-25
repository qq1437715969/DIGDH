import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, in_planes):
        super(Residual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x) + x
        out = self.relu(out)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, f=nn.ReLU(inplace=True)):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            f,
        )

    def forward(self, x):
        return self.conv(x)


class Deconv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, f=nn.ReLU(inplace=True)):
        super(Deconv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            f,
        )

    def forward(self, x):
        return self.conv(x)


class PoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=64, setting=(2, 4, 8, 16, 32)):
        super(PoolingModule, self).__init__()
        self.feature = nn.ModuleList()
        self.setting = setting
        for s in self.setting:
            self.feature.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ))
        self.up = nn.Upsample(32)

    def forward(self, x):
        out_list = []
        for i in range(0, 5):
            out_list.append(self.up(self.feature[i](x)))  # 上采样
        out = torch.cat(out_list, 1)
        return out


class HidingNet(nn.Module):
    def __init__(self, in_channels=6):
        super(HidingNet, self).__init__()

        self.conv_1_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ),
            Conv3x3(32, 64),
            Conv3x3(64, 128),
            Conv3x3(128, 256),
        ])

        self.pooling = PoolingModule(256)

        self.conv_2_list = nn.ModuleList([
            Deconv3x3(512 + 64, 128),
            Deconv3x3(256, 64),
            Deconv3x3(128, 32),
            nn.Sequential(
                nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            ),
        ])

        self.output = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_list = []
        for i in range(0, 4):
            x = self.conv_1_list[i](x)
            x_list.append(x)

        x = self.pooling(x)

        for i in range(0, 4):
            x = self.conv_2_list[i](torch.cat((x, x_list[3 - i]), dim=1))

        x = self.output(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class t(nn.Module):
    def __init__(self):
        super(t, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class HidingNetUDH(HidingNet):
    def __init__(self):
        super(HidingNetUDH, self).__init__(in_channels=3)

    def forward(self, sec, c):
        s_list = []
        for i in range(0, 4):
            sec = self.conv_1_list[i](sec)
            s_list.append(sec)

        sec = self.pooling(sec)

        for i in range(0, 4):
            sec = self.conv_2_list[i](torch.cat((sec, s_list[3 - i]), dim=1))

        sec = self.output(sec)
        return sec + c



if __name__ == '__main__':
    model = HidingNetUDH()
    # model = PoolingModule(256)
    # model = Deconv3x3(6, 3)
    # model = t()

    device = torch.device('cuda:0')
    model = model.to(device)
    # print(model)
    s = torch.randn(1, 3, 256, 256).to(device)
    c = torch.randn(1, 3, 256, 256).to(device)
    # s = torch.randn(1, 256, 32, 32).to(device)
    y = model(s, c)
    print(y.shape)
