# -*- coding: utf-8 -*-
# @Time : 2022/5/16 17:17
# @Author : Shen Junyong
# @File : Base
# @Project : MSAN_Retina
import torch
import torch.nn as nn
import torchvision
class Base_Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self,num_classes):
        super(Base_Model, self).__init__()
        self.fundus_branch = torchvision.models.resnet18(pretrained=True)  
        self.oct_branch = torchvision.models.resnet18(pretrained=True)
        self.fundus_branch.fc = nn.Sequential()     # 移除最后一层全连接层
        self.oct_branch.fc = nn.Sequential()        # 移除最后一层全连接层
        self.decision_branch = nn.Linear(512 * 1 * 2, num_classes)

        #vgg16
        # self.fundus_branch = torchvision.models.vgg16_bn(pretrained=True)  # 移除最后一层全连接层
        # self.oct_branch = torchvision.models.vgg16_bn(pretrained=True)  # 移除最后一层全连接层
        # self.fundus_branch.classifier._modules['6'] = nn.Sequential()
        # self.oct_branch.classifier._modules['6'] = nn.Sequential()
        # self.decision_branch = nn.Linear(4096 * 1 * 2, num_classes)


    def forward(self, fundus_img, oct_img):
    # def forward(self, inputs):
        # fundus_img, oct_img = inputs
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)
        logit = self.decision_branch(torch.cat([b1, b2], 1))

        return logit

if __name__ == '__main__':
    num_classes = 4
    model = Base_Model(4)
    model(torch.ones((1,3,448,448)),torch.ones((1,3,224,224)))