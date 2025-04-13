# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cross_ma_resnet
    author: 12718
    time: 2022/9/14 15:27
    tool: PyCharm
"""
import torch
import torch.nn as nn
from thop import profile
# from torchvision.models import resnet18
from torchvision.models import efficientnet_b3
from torchvision.models import resnet50
from torchvision.models import resnet34,resnet18
import time

class CrossAtt(nn.Module):
    def __init__(self, x_ch, y_ch, dim=128, num_head=4):
        super(CrossAtt, self).__init__()
        self.x_qkv = nn.Conv2d(x_ch, dim*3, 1, 1)
        self.y_qkv = nn.Conv2d(y_ch, dim*3, 1, 1)
        self.num_head = num_head
        h_dim = dim // num_head
        self.gamma = h_dim ** -0.5
        self.hdim = h_dim
        self.proj_x = nn.Conv2d(dim, x_ch, 1, 1)
        self.proj_y = nn.Conv2d(dim, y_ch, 1, 1)

    def forward(self, x, y):
        bs, ch, h_x, w_x = x.size() #1*512*28*28

        qkv_x = self.x_qkv(x).reshape(bs, 3, self.num_head, self.hdim, h_x*w_x).permute(1, 0, 2, 4, 3) #3,bs,num_head:4, h*w:784, h_dim:256
        q_x, k_x, v_x = qkv_x.unbind(0) #1*4*784*256

        bs, ch, h_y, w_y = y.size()

        qkv_y = self.y_qkv(y).reshape(bs, 3, self.num_head, self.hdim, h_y * w_y).permute(1, 0, 2, 4, 3)  # 3,bs,num_head, h*w, h_dim
        q_y, k_y, v_y = qkv_y.unbind(0)

        cross_x = q_x @ k_y.transpose(-2, -1) #bs, num_head, h_x*w_x, h_y*w_y 1*4*784*784
        cross_x = cross_x*self.gamma
        atten_x =torch.softmax(cross_x, dim=-1)
        out_x = atten_x @ v_y #bs, num_head, h_x*w_x, h_dim 1*4*784*256
        out_x = out_x.permute(0, 1, 3, 2).reshape(bs, -1, h_x, w_x) #1*1024*28*28

        cross_y = q_y @ k_x.transpose(-2, -1)  # bs, num_head, h_x*w_x, h_y*w_y
        cross_y = cross_y * self.gamma
        atten_y = torch.softmax(cross_y, dim=-1)
        out_y = atten_y @ v_x  # bs, num_head, h_x*w_x, h_dim
        out_y = out_y.permute(0, 1, 3, 2).reshape(bs, -1, h_y, w_y)

        out_x = self.proj_x(out_x) + x
        out_y = self.proj_y(out_y) + y

        return out_x, out_y

class Attention4D(nn.Module):
    def __init__(self, num_head, dim, dim_k, att_ratio=4.,qkv_bias=True, proj_bias=True,
                 rel_position=None, act_layer=None, downsample=False):
        """
        The efficient Multi-Head Attention in
        "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>
        Args:
            num_head (int): Number of heads
            dim (int): dim of the input
            dim_k (int): dim of each head
            att_ratio (float): expansion rate for the value tensor. (I don't know whether the official code add this)
            qkv_bias (bool): option to switch the bias in qkv linear
            proj_bias (bool): option to switch the bias in proj linear
            rel_position (nn.Module): the relation position module
            act_layer: nonlinear activation function
            downsample(bool): whether downsample the key and value
        """
        super(Attention4D, self).__init__()
        self.dim_k = dim_k
        self.num_head = num_head
        self.scale = dim_k ** -0.5
        act_layer = nn.ReLU if act_layer is None else act_layer
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim_k*num_head, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(dim_k*num_head)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim_k * num_head, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(dim_k * num_head)
        )
        self.d = int(dim_k*att_ratio)
        self.dh = self.num_head * self.d
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1, 1, 0, bias=qkv_bias),
            nn.BatchNorm2d(self.dh)
        )
        self.rel_pos = rel_position if rel_position else nn.Identity()
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, 3, 1, 1, groups=self.dh),
            nn.BatchNorm2d(self.dh)
        )

        self.talking_head1 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.talking_head2 = nn.Conv2d(num_head, num_head, 1, 1, 0)
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1, bias=proj_bias),
            nn.BatchNorm2d(dim)
        )
        self.downsample = nn.Conv2d(dim, dim, 3, 2, 1) if downsample else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x).reshape(B, self.num_head, self.dim_k, -1).transpose(2, 3) #B, Num_Head, N, head_dim
        x = self.downsample(x)
        k = self.k(x).reshape(B, self.num_head, self.dim_k, -1) #B, Num_Head, head_dim, N
        v = self.v(x)
        v = self.v_local(v)
        v = v.reshape(B, self.num_head, self.d, -1).transpose(2, 3) #B, Num_Head, N, head_dim
        att = q @ k #B, Num_Head, N, N
        att = att * self.scale
        att = self.rel_pos(att)
        att = self.talking_head1(att)
        att = torch.softmax(att, dim=-1)
        att = self.talking_head2(att)
        net = att @ v #B, Num_Head, N, Head_Dim
        net = net.transpose(2, 3).reshape(B, -1, H, W)
        net = self.proj(net)
        return net

class UFFN(nn.Module):
    def __init__(self, dim, out_dim, expansion_rate=4.,  act_layer=nn.GELU, use_mid_conv=True):
        """
        Unified FFN introduced at Section 3.1 in
         "Rethinking Vision Transformers for MobileNet Size and Speed"<https://arxiv.org/abs/2212.08059>

        Args:
            dim (int):
            out_dim (int): dimension for output
            expansion_rate (float):  expansion rate
            act_layer (nn.Module): nonlinear function for activation
            use_mid_conv (bool): option to switch the middle depthwise convolution
        """
        super(UFFN, self).__init__()
        if act_layer is None:
            act_layer = nn.GELU
        hidden_dim = int(dim*expansion_rate)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        ) if use_mid_conv else nn.Identity()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )
    def forward(self, x):
        net = self.fc1(x)
        net = self.mid_conv(net)
        net = self.fc2(net)
        return net


class CrossGate(nn.Module):
    def __init__(self, x_ch, y_ch, dim, num_head):
        super(CrossGate, self).__init__()
        # Transformer Block 1
        self.pre_att_x = Attention4D(num_head, x_ch, dim_k=dim)
        self.pre_att_y = Attention4D(num_head, y_ch, dim_k=dim)
        self.mlp_x = UFFN(x_ch, x_ch, use_mid_conv=False)
        self.mlp_y = UFFN(y_ch, y_ch, use_mid_conv=False)

        # Transformer Block 2
        # self.pre_att_x_1 = Attention4D(num_head, x_ch, dim_k=dim)
        # self.pre_att_y_1 = Attention4D(num_head, y_ch, dim_k=dim)
        # self.mlp_x_1 = UFFN(x_ch, x_ch, use_mid_conv=False)
        # self.mlp_y_1 = UFFN(y_ch, y_ch, use_mid_conv=False)

        # Transformer Block 3
        # self.pre_att_x_2 = Attention4D(num_head, x_ch, dim_k=dim)
        # self.pre_att_y_2 = Attention4D(num_head, y_ch, dim_k=dim)
        # self.mlp_x_2 = UFFN(x_ch, x_ch, use_mid_conv=False)
        # self.mlp_y_2 = UFFN(y_ch, y_ch, use_mid_conv=False)

        # CMA block
        self.cross_att = CrossAtt(x_ch, y_ch, dim, num_head)
        self.mlp_x_after = UFFN(x_ch, x_ch, use_mid_conv=False)
        self.mlp_y_after = UFFN(y_ch, y_ch, use_mid_conv=False)

    def forward(self, x, y):
        #Transformer Block 3
        # x = self.pre_att_x_2(x) + x
        # y = self.pre_att_y_2(y) + y
        # x = self.mlp_x_2(x) + x
        # y = self.mlp_y_2(y) + y

        #Transformer Block 2
        # x = self.pre_att_x_1(x) + x
        # y = self.pre_att_y_1(y) + y
        # x = self.mlp_x_1(x) + x
        # y = self.mlp_y_1(y) + y

        #Transformer Block 1
        x = self.pre_att_x(x) + x
        y = self.pre_att_y(y) + y
        x = self.mlp_x(x) + x
        y = self.mlp_y(y) + y

        # CMA block
        x, y = self.cross_att(x, y)
        x = self.mlp_x_after(x)
        y = self.mlp_y_after(y)

        return x, y




class CrossMaResNet(nn.Module):
    def __init__(self, in_ch_x=3, in_ch_y=3, num_classes=4, pretrained=True):
        super(CrossMaResNet, self).__init__()
        self.backbone_x = resnet18(pretrained=pretrained)
        del self.backbone_x.fc
        if in_ch_x != 3:
            self.backbone_x.conv1 = nn.Conv2d(in_ch_x, 64, 7, 2, 3)
        self.backbone_y = resnet18(pretrained=pretrained)
        del self.backbone_y.fc
        if in_ch_y != 3:
            self.backbone_y.conv1 = nn.Conv2d(in_ch_y, 64, 7, 2, 3)

        #Transformer Block and CMA block 3
        # self.cross_att_stage3 = CrossGate(256, 256, 256*1*2, num_head=4)
        # self.cross_fusion_x_3 = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, 1, 0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, 1)
        # )
        # self.cross_fusion_y_3 = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, 1, 0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, 1)
        # )

        # Transformer Block and CMA block 4
        self.cross_att_stage4 = CrossGate(512, 512, 512*1*2, num_head=4)
        self.cross_fusion_x = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1)
        )
        self.cross_fusion_y = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

        #FC
        self.fc = nn.Linear(1024, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Linear(512, num_classes)
        self.fc_y = nn.Linear(512, num_classes)


    def backbone_forward(self, model, input):
        net = model.relu(model.bn1(model.conv1(input)))
        s1 = model.layer1(net)
        s2 = model.layer2(s1)
        s3 = model.layer3(s2)
        # s4 = model.layer4(s3)
        return s3

    def forward(self, x, y):
    # def forward(self, inputs):
    #     x, y = inputs
        x_s3 = self.backbone_forward(self.backbone_x, x)
        y_s3 = self.backbone_forward(self.backbone_y, y)

        # Transformer Block and CMA3 block
        # x_s3, y_s3 = self.cross_att_stage3(x_s3, y_s3)
        # x_s3 = self.cross_fusion_x_3(x_s3)
        # y_s3 = self.cross_fusion_y_3(y_s3)

        x_s4 = self.backbone_x.layer4(x_s3) #B*512*28*28
        y_s4 = self.backbone_y.layer4(y_s3) #B*512*28*28

        #Transformer Block and CMA block
        x_s4, y_s4 = self.cross_att_stage4(x_s4, y_s4) #x_s4, y_s4 = B*512*28*28, B*512*28*28
        x_s4 = self.cross_fusion_x(x_s4) #B*512*28*28
        y_s4 = self.cross_fusion_y(y_s4) #B*512*28*28

        x_s4 = self.avg_pool(x_s4).flatten(1)
        y_s4 = self.avg_pool(y_s4).flatten(1)
        out = torch.cat([x_s4, y_s4], dim=1)
        out = self.fc(out)

        #lose1 and lose2
        out_x = self.fc_x(x_s4)
        out_y = self.fc_y(y_s4)
        return out, out_x, out_y

if __name__ == "__main__":
    x = torch.randn(1, 3, 448, 448).cuda()
    y = torch.randn(1, 3, 448, 448).cuda()
    model = CrossMaResNet().cuda()
    flops, params = profile(model, inputs=(x,))
    print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
    # 开始计时
    # start_time = time.time()
    # out, out_x, out_y = model(x, y)
    # # 结束计时
    # end_time = time.time()
    # 计算推理时间
    # inference_time = end_time - start_time
    # print("Inference time:", inference_time, "seconds")
    # print(out.shape)