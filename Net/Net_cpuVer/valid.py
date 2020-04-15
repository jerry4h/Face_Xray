from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import dataset
import models
from config import config
from config import update_config
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./nnb_adam_lr5e-2_bs32.yaml')

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='./output/BI_dataset/faceXray.pth')

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME + '.get_nnb')(
        config)

    model = torch.nn.DataParallel(model)

    # Data loading code
    valid_dataset = eval('dataset.' + config.DATASET.DATASET + '.' + config.DATASET.DATASET)(
        config.DATASET.ROOT, config.DATASET.TEST_SET, None,
        transforms.Compose([
            transforms.ToTensor()
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    model.eval()

    with torch.no_grad():
        for i, (input, target, _, _) in enumerate(valid_loader):

            # # 输出label
            target = torchvision.utils.make_grid(target).numpy()
            target = np.transpose(target, (1, 2, 0))

            plt.imshow(target)
            plt.show()
            plt.pause(0.5)

            # 输出网络输出
            output = model(input)

            output = torchvision.utils.make_grid(output).numpy()
            output = np.transpose(output, (1, 2, 0))

            plt.imshow(output)
            plt.show()
            plt.pause(0.5)




if __name__ == '__main__':
    main()
