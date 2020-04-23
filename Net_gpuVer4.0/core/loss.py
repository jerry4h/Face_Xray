import torch

import numpy as np
np.set_printoptions(threshold=np.inf)

#debug
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class Loss:
    def __init__(self):
        # loss function
        self.facexrayLoss = torch.nn.BCELoss().cuda()
        self.nncLoss = torch.nn.CrossEntropyLoss().cuda()

    def facexray_loss(self, pred_label, gt_label):
        # face xray loss使用交叉熵
        # 减少维度
        # gt_label = torch.squeeze(gt_label)
        # pred_label = torch.squeeze(pred_label)
        # 为gt_label设置阈值，大于0.5的设为1，使gt像素为0或1
        # 似乎会消除模糊边界？变成不规则边界？
        # gt_label[gt_label >= 0.5] = 1
        # gt_label[gt_label < 0.5] = 0

        # debug
        # gt_label = torchvision.utils.make_grid(gt_label).numpy()
        # gt_label = np.transpose(gt_label, (1, 2, 0))
        # plt.imshow(gt_label)
        # plt.show()
        # plt.pause(0.5)

        return self.facexrayLoss(pred_label, gt_label)

    def ncc_loss(self, pred_label, gt_label):

        return self.nncLoss(pred_label, gt_label)