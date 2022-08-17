import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            # 36个embedding网，1个超网
            # 对于36个不同的z_list，1个参数生成网络
            # 其中所有参数都是通过torch.randn生成，然后通过torch.fmod转换为二值

            # embedding网，控制参数shape，首先生成 z_list ,即 z_num * z_num 个长度为 z_dim 的0-1Tensor，然后将z_list交给超网
            # 超网用来生成参数，对于每一个长为 z_dim 的Tensor，将其转换为[16,16,3,3]的参数组，其中转换的矩阵参数也是0-1Tensor
            # 然后enbedding网在 dim=1 维度上进行concat，形成一个Conv2D的参数

            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x
