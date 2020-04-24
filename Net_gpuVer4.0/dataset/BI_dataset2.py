import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

class BI_dataset2(data.Dataset):

    def __init__(self, root, image_set_name, list_name = None, Transform = transforms.ToTensor()):
        super(BI_dataset2, self).__init__()
        self.root = os.path.join(root, image_set_name)
        if list_name == None:
            raise NotImplementedError
            # list_name = image_set_name + '.txt'
            # self.list_path = os.path.join(self.root, list_name)
        else:
            list_path = os.path.join(root, list_name)
        self.transform = Transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for i, name in enumerate(img_ids):
            img_file = os.path.join(self.root, "%s.jpg" % name)
            label_file = os.path.join(self.root, "%s_label.jpg" % name)
            cls_label = 1

            self.files.append({
                "img": img_file,
                "label": label_file,
                "cls_label": cls_label,
                "name": name
            })
    
    def loadTrue(self, root, image_set_name, list_name = None):
        if list_name is None:
            raise NotImplementedError
        else:
            list_path = os.path.join(root, list_name)
        img_ids = [i_id.strip() for i_id in open(list_path)]

        for i, name in enumerate(img_ids):
            img_file = os.path.join(root, image_set_name, "%s.jpg" % name)
            label_file = os.path.join(root, "0.jpg")
            cls_label = 0

            self.files.append({
                "img": img_file,
                "label": label_file,
                "cls_label": cls_label,
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
        cls_label = int(datafiles["cls_label"])

        size_origin = image.size  # W * H

        # 将PIL image转换为Tensor，Transform参数一般为 transforms.ToTensor()
        if self.transform is not None:
            image = self.transform(image)
            image = self.normalize(image)
            label = self.transform(label)

        return image, label, cls_label, np.array(size_origin), name


# 测试
if __name__ == '__main__':

    DATA_DIRECTORY = '/nas/hjr'
    DATA_LIST_PATH = 'celebrityBlended'
    Batch_size = 4

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    dataset = BI_dataset2('/nas/hjr', 'celebrityBlended', 'train1.txt', Transform=transforms)
    dataset.loadTrue('/nas/hjr', 'celebritySelect0', 'train0.txt')
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

