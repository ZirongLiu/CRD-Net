# -*- coding: utf-8 -*-
# @Time : 2022/5/11 10:25
# @Author : Shen Junyong
# @File : model_list
# @Project : MSAN_Retina
from model.MMCNN import *
from model.Base import Base_Model
from model.CRDNet import CrossMaResNet
from model.MSAN import DRT_18_18

def create_model(model_name, args):
    if model_name == 'Two_stream_Cat_resnet18':
        model = Two_stream_Cat_resnet18(num_classes = args.num_classes)
        return model
    elif model_name == 'Two_stream_Cat_resnet34':
        model = Two_stream_Cat_resnet34(num_classes = args.num_classes)
        return model
    elif model_name == 'Two_stream_Cat_resnet50':
        model = Two_stream_Cat_resnet50(num_classes = args.num_classes)
        return model
    elif model_name == 'Two_stream_VOTE_resnet18':
        model = Two_stream_VOTE_resnet18(num_classes = args.num_classes)
        return model
    elif model_name == 'MM-CNN':
        model = Base_Model(num_classes=args.num_classes)
        return model
    elif model_name == 'CRD-Net':
        model = CrossMaResNet(num_classes=args.num_classes)
        return model
    elif model_name == 'MSAN':
        model = DRT_18_18(num_classes=args.num_classes)
        return model
    else:
        assert False, 'No Model'