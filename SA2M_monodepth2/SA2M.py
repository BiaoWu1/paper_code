import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AC(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(AC, self).__init__()
        # self.gate_channels = gate_channels
        # self.mlp = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        #     nn.BatchNorm1d(gate_channels // reduction_ratio)
        #     nn.ReLU(),
        #     nn.Linear(gate_channels // reduction_ratio, gate_channels)
        #     )

        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module('gate_c_fc_%d'%(i+1), nn.Linear(gate_channels[i+1], gate_channels[i]))
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
    def forward(self, in_tensor):
        avg_pool = F.adaptive_avg_pool2d( in_tensor, (1, 1))
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(avg_pool)

class ALS(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(ALS, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_2d_11', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1, stride=1, padding=0))
        self.gate_s.add_module( 'gate_s_conv_3', nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, stride=1, padding=1))
        self.gate_s.add_module( 'gate_s_conv_12', nn.Conv2d(gate_channel//reduction_ratio, gate_channel, kernel_size=1, stride=1, padding=0))
        self.gate_s.add_module( 'gate_s_relu_rule',nn.ReLU())
        self.gate_s.add_module( 'gate_s_bn', nn.BatchNorm2d(gate_channel))
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class AGS(nn.Module):
    def __init__(self, channel):
        super(AGS, self).__init__()
        self.inter_channel = channel // 8
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class SA2M(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SA2M, self).__init__()
        self.AC = AC(gate_channels, reduction_ratio)
        self.ALS = ALS(gate_channels, reduction_ratio)
        self.AGS = AGS(gate_channels)
    def forward(self, x):
        AC_out = self.AC(x)
        GAP_ALS_out = F.adaptive_avg_pool2d(self.ALS(x), (1, 1))
        GAP_AGS_out = F.adaptive_avg_pool2d(self.AGS(x), (1, 1))
        max_of_AC_and_ALS = torch.max(AC_out, GAP_ALS_out)
        max_of_final = torch.max(max_of_AC_and_ALS, GAP_AGS_out)
        #mean
        mean_res = torch.mean(max_of_final, dim = 1, keepdim = True)
        #sofmax
        softmax_result = F.softmax(mean_res, dim = 0)
        x_out = softmax_result * x
        return x_out
