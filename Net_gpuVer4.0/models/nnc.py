
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class NNC(nn.Module):

    def __init__(self):
        super(NNC, self).__init__()

        # 全局平均池化层
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，分成两类，是否伪造
        self.linear = nn.Linear(1 * 1 * 1, 2)

    def forward(self, x):
        # 全局平均池化层 : [batch, 1, 1, 1]
        x = self.pool(x)
        # 为了进行FC，把原输出展成FC可接受的一维data [batch, 1*1*1]
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        return probs


    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print('[MODEL] loading from %s' %pretrained)
            # 用cpu加载模型参数时
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            # 锁定原hrnet的参数
            for k, v in self.named_parameters():
                if 'upsample_modules' in k or 'output_modules' in k:
                    continue
                else:
                    v.requires_grad = False


def get_nnc(config, **kwargs):
    model = NNC()
    model.init_weights(config.TEST.NNC_FILE)
    return model
