import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
from torch.utils import data

from dataset.blend_utils.faceBlending import Blender
from dataset.aug_trans.aug_trans import Augmentator, data_transform


class BI_dataset_dynamic_aug(data.Dataset):
    """读取和测试的 BI 数据集，标签写在 list.txt 里，采用 albumentations 做增强
    
    读取参数：目录、具体路径与标签名（如果是1，应该有对应路径+_label.jpg的标签）
    root 目录下应该有 0.jpg 的全黑图片，对应标签为 0 的 xray
    对于 test set，返回的 xray label 就是 cls_label
    """

    def __init__(self, root, list_name, mode='train', Transform='simple'):
        super(BI_dataset_dynamic_aug, self).__init__()
        if mode not in ['train'] or \
            Transform not in ['simple', 'pixel']:
            raise NotImplementedError()

        self.root = root
        list_path = os.path.join(root, list_name)  # landmarks
        
        self.mode = 'train'
        
        if Transform == 'pixel':
            self.pixel_aug = Augmentator('pixel_aug')
            self.spatial_aug = None  # TODO
        elif Transform == 'simple':
            self.pixel_aug = Augmentator('simple')
            self.spatial_aug = None
        else:
            raise NotImplementedError(Transform)

        self.blender = Blender(
            ldmPath=list_path, dataPath=root,
            topk=100, selectNum=1, gaussianKernel=[31,63], gaussianSigma=[7, 15], loader='cv',
            pixel_aug=self.pixel_aug, spatial_aug=self.spatial_aug, aug_at_load=True
        )
        
        self.trans_image = data_transform(normalize=True)
        self.trans_xray = data_transform(normalize=False)

    def __len__(self):

        return len(self.blender)

    def __getitem__(self, index):

        image, blended, xray = None, None, None
        for blended_, xray_ in self.blender.blend_i(index):
            image = self.blender.img_loader(index, do_aug=True)
            blended, xray = blended_, xray_

        size_origin = image.size  # W * H

        # 将opencv image转换为Tensor
        image = self.trans_image(image=image)['image']
        blended = self.trans_image(image=blended)['image']
        xray = self.trans_xray(image=xray)['image']
        xray = xray[0].unsqueeze(0)

        return [image, blended], [torch.zeros_like(xray), xray], [0, 1], \
            np.array(size_origin), 'unkown Path'


# 测试
if __name__ == '__main__':

    ROOTPATH = 'D:/Dataset/blendDatasetTest/generatorTop10'  # '/mnt/hjr/FF++c23/original/generator'
    LISTNAME = 'D:/Dataset/blendDatasetTest/originalC23X100kLmTop10.txt'
    Batch_size = 4

    dataset = BI_dataset_dynamic_aug(root=ROOTPATH, list_name = LISTNAME, mode='train', Transform='pixel')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)

    plt.ion()

    for i, data in enumerate(dataloader):
        imgs, labels, cls_label, _, _ = data
        imgs, labels, cls_label = torch.cat(imgs), torch.cat(labels), torch.cat(cls_label)
        # import pdb
        # pdb.set_trace()

        # 减少第0个维度
        # imgs = imgs.squeeze(0)
        # labels = labels.squeeze(0)

        # 把所有图像拼在一起
        img = torchvision.utils.make_grid(imgs).numpy()
        labels = torchvision.utils.make_grid(labels).numpy()

        imgs = np.transpose(img, (1, 2, 0))
        labels = np.transpose(labels, (1, 2, 0))
        pairs = np.vstack([imgs, labels])

        plt.imshow(pairs)
        import pdb
        pdb.set_trace()
        plt.show()
        plt.pause(0.5)

