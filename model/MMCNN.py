import torch.nn as nn
import torch
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Two_Stream_Resnet(nn.Module):
    def __init__(
            self,
            Classifier_mode,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Two_Stream_Resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv11 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.conv21 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(self.inplanes)

        self.layer11 = self._make_layer(block, 64, layers[0])
        self.layer12 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer13 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer14 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #重置
        self.inplanes = 64
        self.layer21 = self._make_layer(block, 64, layers[0])
        self.layer22 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer23 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer24 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Classifier_mode = Classifier_mode
        #self.fc = nn.Linear(512 * block.expansion*2, num_classes)
        if self.Classifier_mode == 'CAT_MLP':
            self.Classifier = CAT_MLP(512 * block.expansion*2, num_classes)
        elif self.Classifier_mode == 'VOTE':
            self.Classifier = None

        self.fc_1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc_2 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        B, _, _, _ = x1.size()
        x1 = self.conv11(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer11(x1)
        x1 = self.layer12(x1)
        x1 = self.layer13(x1)
        x1 = self.layer14(x1)
        x1 = self.avgpool(x1)

        x2 = self.conv21(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer21(x2)
        x2 = self.layer22(x2)
        x2 = self.layer23(x2)
        x2 = self.layer24(x2)
        x2 = self.avgpool(x2)

        logits_1 = self.fc_1(x1.view(B,-1))
        logits_2 = self.fc_2(x2.view(B,-1))
        if self.Classifier_mode == 'CAT_MLP':
            x = self.Classifier(x1,x2)
        elif self.Classifier_mode == 'Vote':
            x = 0.5*logits_1+0.5*logits_2
        return x, logits_1, logits_2

class CAT_MLP(nn.Module):
    def __init__(self, input, num_classes):
        super(CAT_MLP, self).__init__()
        self.fc = nn.Linear(input, num_classes)
    def forward(self, x1, x2):
        B,_,_,_ = x1.size()
        x = self.fc(torch.cat([x1.view(B,-1), x2.view(B,-1)], dim=1))
        return x

#Cat two stream
def Two_stream_Cat_resnet18(**kwargs):
    model_18 = Two_Stream_Resnet('CAT_MLP', BasicBlock, [2, 2, 2, 2], **kwargs)
    return model_18

def Two_stream_Cat_resnet34(**kwargs):
    model_34 = Two_Stream_Resnet('CAT_MLP', BasicBlock, [3, 4, 6, 3], **kwargs)
    return model_34

def Two_stream_Cat_resnet50(**kwargs):
    model_50 = Two_Stream_Resnet('CAT_MLP', Bottleneck, [3, 4, 6, 3], **kwargs)
    return model_50

#VOTE two stream
def Two_stream_VOTE_resnet18(**kwargs):
    model_18 = Two_Stream_Resnet('VOTE', BasicBlock, [2, 2, 2, 2], **kwargs)
    return model_18

def Two_stream_VOTE_resnet34(**kwargs):
    model_34 = Two_Stream_Resnet('VOTE', BasicBlock, [3, 4, 6, 3], **kwargs)
    return model_34

def Two_stream_VOTE_resnet50(**kwargs):
    model_50 = Two_Stream_Resnet('VOTE', Bottleneck, [3, 4, 6, 3], **kwargs)
    return model_50

