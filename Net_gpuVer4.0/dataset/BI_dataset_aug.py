import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
from torch.utils import data
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensor


def cv_loader(path, gray=False):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except IOError:
        print('Cannot load image ' + path)

def easy_transform():
    t = Compose([
        Resize(256, 256)
    ])
    return t


def strong_aug_pixel(p=.5):
    print('[DATA]: strong aug pixel')

    from albumentations import (
    # HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, MultiplicativeNoise,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, JpegCompression, CLAHE)

    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True),
            JpegCompression(quality_lower=39, quality_upper=80)
        ], p=0.2),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def data_transform(size=256, normalize=True):
    if normalize:
        t = Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor()
        ])
    else:
        t = Compose([
            Resize(size, size),
            ToTensor()
        ])
    return t


class BI_dataset_aug(data.Dataset):
    """读取和测试的 BI 数据集，标签写在 list.txt 里，采用 albumentations 做增强
    
    读取参数：目录、具体路径与标签名（如果是1，应该有对应路径+_label.jpg的标签）
    root 目录下应该有 0.jpg 的全黑图片，对应标签为 0 的 xray
    对于 test set，返回的 xray label 就是 cls_label
    """

    def __init__(self, root, list_name, mode='train', Transform='easy'):
        super(BI_dataset_aug, self).__init__()
        if mode not in ['train', 'valid', 'test'] or\
            Transform not in ['easy', 'strong_pixel']:
            raise NotImplementedError()

        self.mode = 'train'
        self.root = root
        if mode == 'test':
            self.mode = 'test'

        list_path = os.path.join(root, list_name)
        
        if Transform == 'easy':
            self.trans_aug = easy_transform()
        elif Transform == 'strong_pixel':
            self.trans_aug = strong_aug_pixel()
        self.trans_data = data_transform(normalize=True)
        self.trans_label = data_transform(normalize=False)
    
        labels, img_paths = [], []
        with open(list_path, 'r') as f:
            for line in f:
                label, img_path = line.strip().split()
                labels.append(int(label))
                img_paths.append(img_path)

        self.files = []
        self.cnts = [0, 0]

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
            self.cnts[label] += 1
            
            cls_label = label
            self.files.append({
                "img": img_file,
                "label": label_file,
                "cls_label": cls_label,
                "name": name
            })
        print('[DATA] label images for dataset %s: %d:%d' %(list_path, *self.cnts))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        label = None
        # 将图片和label读出。“L”表示灰度图，也可以填“RGB”
        name = datafiles["name"]
        image = cv_loader(datafiles["img"])
        cls_label = int(datafiles["cls_label"])
        if self.mode != 'test':
            label = cv_loader(datafiles["label"], gray=True)
            # label = np.expand_dims(label, 0)
        else:
            label = cls_label

        size_origin = image.size  # W * H

        # 将opencv image转换为Tensor
        image = self.trans_aug(image=image)['image']
        image = self.trans_data(image=image)['image']
        if self.mode != 'test':  # 测试下没有 xray 的 label 标签
            label = self.trans_label(image=label)['image']
            label = label.unsqueeze(0)

        return image, label, cls_label, np.array(size_origin), name


# 测试
if __name__ == '__main__':

    ROOTPATH = '/nas/hjr/FF++c23/original'
    LISTNAME = 'selected10kTrain.txt'
    Batch_size = 4

    dataset = BI_dataset_aug(root=ROOTPATH, list_name = LISTNAME, mode='train', Transform='easy')
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

