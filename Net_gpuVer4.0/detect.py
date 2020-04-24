import os.path as osp
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from config import config
from config import update_config
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Detect for Images')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='./nnb_adam_lr5e-2_bs32.yaml')
    parser.add_argument('--nnb', help='NNB model path', type=str,
                        default='D:/Dataset/faceXray_10.pth')
    parser.add_argument('--nnc', help='NNC model path', type=str, default='D:/Dataset/nnc10.pth')
    parser.add_argument('--root', help='images root path', type=str, default='D:/Dataset/ai_pictures/x25.mp4')
    parser.add_argument('--list', help='image id list', type=str, default='D:/Dataset/ai_pictures/x25.txt')

    args = parser.parse_args()
    update_config(config, args, simple=True)
    return args


def loadModel(args):
    nnb = models.nnb.get_nnb(config)
    nnc = models.nnc.get_nnc(False)
    nnb.load_state_dict(torch.load(args.nnb, map_location='cpu'))
    nnc.load_state_dict(torch.load(args.nnc, map_location='cpu'))
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

def preProcessPIL(imgPath, transform):
    image = Image.open(imgPath).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)

def getPredictions(img, nnb, nnc):
    with torch.no_grad():
        xray = nnb(img)
        prob = nnc(xray)
    return xray, prob

def visualize(img, xray, prob):
    # import pdb
    # pdb.set_trace()
    xray_color = torch.cat((xray, xray, xray), 1)
    grid_tensor = torch.cat((img, xray_color), dim=0)
    target = torchvision.utils.make_grid(grid_tensor).numpy()
    target = np.transpose(target, (1, 2, 0))
    plt.imshow(target)
    print(prob)
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    nnb, nnc = loadModel(args)

    if args.list:
        imgNames = [i.rstrip('\n')+'.jpg' for i in open(args.list, 'r')]
    else:
        imgNames = os.listdir(args.root)
    for imgName in imgNames:
        img = preProcessPIL(osp.join(args.root, imgName), transform)
        xray, prob = getPredictions(img, nnb, nnc)
        visualize(img, xray, prob)

