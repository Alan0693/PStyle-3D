import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d')
from training.networks_stylegan2 import FullyConnectedLayer, MappingNetwork, DiscriminatorEpilogue


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StarDiscriminator(nn.Module):
    def __init__(self, img_size=512, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out


class StarDiscriminator_pose(nn.Module):
    def __init__(self, img_size=512, max_conv_dim=512, c_dim=25):
        super().__init__()
        self.c_dim = c_dim
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        # blocks += [FullyConnectedLayer(512 * (2 ** 2), 512, activation='lrelu')]
        # blocks += [FullyConnectedLayer(512, 512, activation='linear')]
        self.main = nn.Sequential(*blocks)
        self.b4 = DiscriminatorEpilogue(512, cmap_dim=512, resolution=4, img_channels=3)
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=512, num_ws=None, w_avg_beta=None)

    def forward(self, x, c):
        out = self.main(x)
        # out = out.view(out.size(0), -1)  # (batch, num_domains)
        cmap = None
        img = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)   # 8层MLP: [1, 25] -> [1, 512]
        out = self.b4(out, img, cmap)
        return out


def compute_d_loss(nets, x_real, x_fake, c):
    # with real images
    x_real.requires_grad_()
    out = nets(x_real, c)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    out = nets(x_fake, c)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + 1 * loss_reg
    return loss


def compute_g_loss(nets, x_fake):
    # adversarial loss
    out = nets(x_fake, c)
    loss_adv = adv_loss(out, 1)

    loss = loss_adv
    return loss


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


if __name__ == "__main__":

    # BlendGAN
    device = 'cuda:1'
    net = StarDiscriminator_pose().to(device)
    print("Information")
    for name, parms in net.named_parameters():
        print("-->name:", name)
        # print("para:", parms)
        print("-->grad_requirs", parms.requires_grad)
    x_real = torch.randn(1, 3, 512, 512).to(device)
    c_cam = torch.randn(1, 25).to(device)
    x_fake = torch.randn(1, 3, 512, 512).to(device)
    d_loss = compute_d_loss(net, x_real, x_fake, c_cam)
    # g_loss = compute_g_loss(net, x_fake)
    print(d_loss)
    # print(g_loss)