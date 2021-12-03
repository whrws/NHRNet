import torch.nn as nn
import torch.nn.functional as F
from model.subnet import GCN_Dehaze
import torch

class GF_layer(nn.Module):
    def __init__(self):
        super(GF_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)

        condition_conv1 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        sift_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

        self.pa = nn.Sequential(
            nn.Conv2d(128, 128 // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // 8, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, depth):

        depth_condition = self.condition_conv(depth)

        sifted_feature = self.sift_conv(depth_condition)

        y = self.pa(torch.cat((x, sifted_feature), 1))

        y1, y2 = torch.split(y, 1, dim=1)

        gated_feature = x * y1 + sifted_feature * y2

        return gated_feature

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.r = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                               nn.ReLU(),
                               nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                               nn.AdaptiveAvgPool2d(1),
                               nn.Sigmoid()
                               )

        self.GF = GF_layer()

        self.refine = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.conv_1 = nn.Sequential(nn.Conv2d(16, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.res1 = ResidualBlock(16)

        self.conv_2 = nn.Sequential(
                                    nn.Conv2d(64, 16, 3, padding=1, bias=False),
                                    )

        self.dehaze = GCN_Dehaze()

        self.start = nn.Conv2d(3, 16, 3, padding=1, bias=False)

    def forward(self, haze):

        r = self.r(haze)

        a, b = torch.split(r, 4, dim=1)

        a1, a2, a3, a4 = torch.split(a, 1, dim=1)

        b1, b2, b3, b4 = torch.split(b, 1, dim=1)

        b1 = b1 + 1

        b2 = b2 + 1

        b3 = b3 + 1

        b4 = b4 + 1

        H_1 = torch.clamp((haze + a1 * (torch.pow(haze + 1e-8, b1) - haze)), 0, 1)
        H_2 = torch.clamp((H_1 + a2 * (torch.pow(H_1 + 1e-8, b2) - H_1)), 0, 1)
        H_3 = torch.clamp((H_2 + a3 * (torch.pow(H_2 + 1e-8, b3) - H_2)), 0, 1)
        H_4 = torch.clamp((H_3 + a4 * (torch.pow(H_3 + 1e-8, b4) - H_3)), 0, 1)
        contrast_feature = torch.cat((H_1, H_2, H_3, H_4), 1)

        haze_process = self.res1(self.start(haze))

        haze_fea = self.conv_1(haze_process)

        haze_fea = self.GF(haze_fea, contrast_feature)

        H_G = self.conv_2(haze_fea)

        H_F = self.refine(H_G)

        dehaze = self.dehaze(H_F, H_G, haze_process)

        return dehaze







