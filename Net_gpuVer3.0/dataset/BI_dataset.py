import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

class BI_dataset(data.Dataset):

    def __init__(self, root, image_set_name, list_name = None, Transform = transforms.ToTensor()):
        super(BI_dataset, self).__init__()
        self.root = os.path.join(root, image_set_name)
        if list_name == None:
            list_name = image_set_name + '.txt'
            self.list_path = os.path.join(self.root, list_name)
        else:
            self.list_path = os.path.join(self.root, list_name)
        self.transform = Transform
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = os.path.join(self.root, "data/%s.jpg" % name)
            label_file = os.path.join(self.root, "label/%s.jpg" % name)

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        # 将图片和label读出。“L”表示灰度图，也可以填“RGB”
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('L')

        size_origin = image.size  # W * H

        # 将PIL image转换为Tensor，Transform参数一般为 transforms.ToTensor()
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label, np.array(size_origin), name


# 测试
if __name__ == '__main__':

    DATA_DIRECTORY = '../data/train'
    DATA_LIST_PATH = '../data/train/train.txt'
    Batch_size = 4
    dataset = BI_dataset('../data', 'train', None, Transform = transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=False)

    plt.ion()

    for i, data in enumerate(dataloader):

        imgs, labels, _, _ = data

        #print(imgs)

        print(imgs.shape, labels.shape)

        # 减少第0个维度
        imgs = imgs.squeeze(0).numpy()
        labels = labels.squeeze(0).numpy()

        # 把所有图像拼在一起
        # img = torchvision.utils.make_grid(imgs).numpy()
        # labels = torchvision.utils.make_grid(labels).numpy()

        imgs = np.transpose(imgs, (1, 2, 0))
        labels = np.transpose(labels, (1, 2, 0))

        plt.imshow(imgs)
        plt.show()
        plt.pause(0.5)

