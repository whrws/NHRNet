import torch.nn.functional as F
from torchvision.models import resnet
from model.GCN_SGR_CGR import GCN_SGR, GCN_CGR
import torch
import torch.nn as nn


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


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):

        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True), )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class GCN_Dehaze(nn.Module):
    def __init__(self):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(GCN_Dehaze, self).__init__()

        base = resnet.resnet50(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(256, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(1024, 512, 3, 2, 1, 1)
        self.decoder4 = Decoder(2048, 1024, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, 16, 2, 2, 0)
        self.lsm = nn.ReLU()
        self.upsample = F.interpolate
        self.conv3 = nn.Sequential(ResidualBlock(48),
                                   nn.Conv2d(48, 3, 3, 1, 1),
                                   nn.ReLU(inplace=True), )

        self.SP = GCN_SGR(in_channel=512, state_channel=512, node_num=256)   # spatial graph
        self.CA = GCN_CGR(in_channel=512, state_channel=256, node_num=512)   # channel graph

    def forward(self, x, z, f):

        x = self.in_block(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        e2_1 = self.SP(e2)
        e2_2 = self.CA(e2)
        e2_3 = e2_1 + e2_2

        e3 = self.encoder3(e2_3)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.upsample(self.decoder4(e4), size=e3.shape[2:])
        d3 = e2 + self.upsample(self.decoder3(d4), size=e2.shape[2:])
        d2 = e1 + self.upsample(self.decoder2(d3), size=e1.shape[2:])
        d1 = x + self.upsample(self.decoder1(d2), size=x.shape[2:])

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        y = self.lsm(y)

        y = torch.cat((y, z, f), 1)
        y = self.conv3(y)
        return y
    
