import os.path as osp
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

from aug import select_aug, strong_aug_pixel, data_transform

from config import config
from config import update_config
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Detect for Images')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./hrw18_adam_lr5e-2_bs32.yaml')
    parser.add_argument('--testNNB',
                        help='testNNB',
                        type=str,
                        default='D:/Dataset/HRw18NNC/nnb0.pth')
    parser.add_argument('--testNNC',
                        help='testNNC',
                        type=str,
                        default='D:/Dataset/HRw18NNC/nnc0.pth')
    parser.add_argument('--list', help='image id list', type=str, default='D:/Dataset/augTest/*.jpg')

    args = parser.parse_args()
    update_config(config, args, simple=True)
    return args


def loadModel(args):
    nnb = models.nnb.get_nnb(config)
    # nnb = models.fcn.get_fcn()
    nnc = models.nnc.get_nnc(config)
    # nnb.load_state_dict(torch.load(args.testNNB, map_location='cpu'))
    # nnc.load_state_dict(torch.load(args.testNNC, map_location='cpu'))
    nnb.eval()
    nnc.eval()
    return nnb, nnc

def preProcess(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # import pdb
    # pdb.set_trace()
    img = img.astype('float')
    
    img /= 255
    img = cv2.resize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    # TODO: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img = (img - 0.45) / 0.225
    return img


def preProcessCV(imgPath, trans_aug, trans_data, gray=False):
    try:
        with open(imgPath, 'rb') as f:
            img = cv2.imread(imgPath)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except IOError:
        print('Cannot load image ' + imgPath)
    img_aug = trans_aug(image=img)['image']
    tensor = trans_data(image=img_aug)['image']
    return tensor, img_aug, img


def preProcessPIL(imgPath, transform):
    image = Image.open(imgPath).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)


def getPredictions(img, nnb, nnc):
    with torch.no_grad():
        xray = nnb(img)
        logit = nnc(xray)
        prob = torch.softmax(logit, dim=1)

    return xray, prob


def visualize(img, xray, prob, vis=False):
    # xray = F.interpolate(xray, img.size()[2:], mode='bilinear', align_corners=True)
    xray_color = torch.cat((xray, xray, xray), 1)
    # grid_tensor = torch.cat((img, xray_color), dim=0)
    target = torchvision.utils.make_grid(xray_color).numpy()
    target = np.transpose(target, (1, 2, 0))
    img = img / 255.
    print(prob)
    concated = np.hstack([img, target])
    if vis:
        plt.imshow(concated)
    return concated

if __name__ == '__main__':
    args = parse_args()

    # ATYPE, PARAMS = 'JpegCompression', [90, 70, 50, 30, 1]
    # ATYPE, PARAMS = 'Blur', [1, 10, 20, 30, 40]
    # ATYPE, PARAMS = 'Downscale', [0.99, 0.9, 0.7, 0.5, 0.3]
    # ATYPE, PARAMS = 'CLAHE', [1.1, 1.5, 2.0, 4.0, 8.0]
    # ATYPE, PARAMS = 'HueSaturationValue', [1, 10, 20, 30, 40]
    # ATYPE, PARAMS = 'RandomBrightnessContrast', [0.01, 0.1, 0.2, 0.4, 0.6]
    # ATYPE, PARAMS = 'IAAAdditiveGaussianNoise', [0, 1, 3, 5, 7, 9]
    # ATYPE, PARAMS = 'GaussNoise', [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    # ATYPE, PARAMS = 'GaussianBlur', [1, 3, 7, 15, 31]
    # ATYPE, PARAMS = 'MedianBlur', [1, 3, 7, 9, 15]
    ATYPE, PARAMS = 'MotionBlur', [4, 7, 11, 19, 31]

    h = len(PARAMS)

    nnb, nnc = loadModel(args)

    imgPaths = glob.glob(args.list)

    concated = []
    for imgPath in imgPaths:
        concated_i = []
        for i, param in enumerate(PARAMS):
            trans_aug = select_aug(atype=ATYPE, param=param)
            trans_data = data_transform()
            tensor, img_aug, img = preProcessCV(imgPath, trans_aug, trans_data)
            xray, prob = getPredictions(tensor.unsqueeze(0), nnb, nnc)
            # plt.subplot(h, 1, i+1)
            concated_i.append(visualize(img_aug, xray, prob))
        concated_i = np.vstack(concated_i)
        concated.append(concated_i)
    concated = np.hstack(concated)
    plt.imshow(concated)
    plt.xlabel(ATYPE + ': {}'.format(PARAMS))
    plt.show()

