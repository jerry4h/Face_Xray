import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
from torch.utils import data

from dataset.blend_utils.faceBlending import Blender
from dataset.aug_trans.aug_trans import Augmentator, data_transform


class BI_dataset_saved_aug(data.Dataset):
    """读取和测试的 BI 数据集，标签写在 list.txt 里，采用 albumentations 做增强
    
    读取参数：目录、具体路径与标签名（如果是1，应该有对应路径+_label.jpg的标签）
    root 目录下应该有 0.jpg 的全黑图片，对应标签为 0 的 xray
    对于 test set，返回的 xray label 就是 cls_label
    """

    def __init__(self, root, list_name, mode='train', Transform='simple'):
        super(BI_dataset_saved_aug, self).__init__()
        if mode not in ['train', 'valid', 'test'] or\
            Transform not in ['simple', 'pixel']:
            raise NotImplementedError()

        self.mode = 'train'
        if mode == 'test':
            self.mode = 'test'
        
        self.root = root
        list_path = os.path.join(root, list_name)

        self._parse_data(list_path)

        
        if Transform == 'simple':
            self.pixel_aug = Augmentator('simple')
            self.spatial_aug = None
        elif Transform == 'pixel':
            self.pixel_aug = Augmentator('pixel_aug')
            self.spatial_aug = None
        else:
            raise NotImplementedError(Transform)

        # 只为用到 Blender.img_loader，为了保持与训练一致
        self.blender = Blender(
            ldmPath=None, dataPath=None,
            topk=100, selectNum=1, gaussianKernel=[31,63], gaussianSigma=[7, 15], loader='cv',
            pixel_aug=self.pixel_aug, spatial_aug=self.spatial_aug
        )

        self.trans_image = data_transform(normalize=True)
        self.trans_xray = data_transform(normalize=False)
    
    def _parse_data(self, list_path):

        labels, img_paths = [], []
        with open(list_path, 'r') as f:
            for line in f:
                label, img_path = line.strip().split()
                labels.append(int(label))
                img_paths.append(img_path)

        self.files = []
        cnts = [0, 0]

        for label, name in zip(labels, img_paths):
            img_file = os.path.join(self.root, name)
            if self.mode == 'test':
                label_file = None
            elif label == 0:
                label_file = os.path.join(self.root, "0.jpg")
            elif label == 1:
                nameRoot, ext = os.path.splitext(name)
                label_file = os.path.join(self.root, "%s_label%s" % (nameRoot, ext))
            else:
                raise NotImplementedError('label: %d' % label)
            cnts[label] += 1
            
            cls_label = label
            self.files.append({
                "img": img_file,
                "label": label_file,
                "cls_label": cls_label,
                "name": name
            })
        print('[DATA] label images for dataset %s: %d:%d' %(list_path, *cnts))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        name = datafiles["name"]
        image = self.blender.img_loader(datafiles["img"])
        cls_label = int(datafiles["cls_label"])

        xray = None
        if self.mode != 'test':
            xray = self.blender.img_loader(datafiles["label"], gray=True)
            # xray = np.expand_dims(xray, 0)
        else:
            xray = cls_label

        size_origin = image.size  # W * H

        # 将opencv image转换为Tensor
        image = self.trans_image(image=image)['image']
        if self.mode != 'test':  # 测试下没有 xray 的 label 标签
            xray = self.trans_xray(image=xray)['image']
            xray = xray.unsqueeze(0)

        return image, xray, cls_label, np.array(size_origin), name


# 测试
if __name__ == '__main__':

    ROOTPATH = '/nas/hjr/FF++c23/original'
    LISTNAME = 'selected10kTrain.txt'
    Batch_size = 4

    dataset = BI_dataset_saved_aug(root=ROOTPATH, list_name = LISTNAME, mode='train', Transform='simple')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True)

    plt.ion()

    for i, data in enumerate(dataloader):
        imgs, labels, cls_label, _, _ = data
        import pdb
        pdb.set_trace()

        # 减少第0个维度
        # imgs = imgs.squeeze(0)
        # labels = labels.squeeze(0)

        # 把所有图像拼在一起
        img = torchvision.utils.make_grid(imgs).numpy()
        labels = torchvision.utils.make_grid(labels).numpy()

        imgs = np.transpose(img, (1, 2, 0))
        labels = np.transpose(labels, (1, 2, 0))

        plt.imshow(imgs)
        plt.show()
        plt.pause(0.5)

