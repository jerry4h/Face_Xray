
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import random
import numpy as np
import cv2
import os
import json
import glob

# import sys; sys.path.insert(0, '../')

from dataset.aug_trans.aug_trans import Augmentator, data_transform

ORIGINAL_METHODS = [
    'original'
]

TRAIN_METHODS = [
    'Face2Face'
]
VALID_METHODS = [
    'Face2Face'
]
TEST_METHODS = [
    'FaceSwap',
    'Deepfakes'
]

DATASET_PATHS = {
    'original': 'original_sequences',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    # 'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
# COMPRESSION = ['c0', 'c23', 'c40']
IMAGE_PATH = 'calculated_xray'


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


class Xray_FFpp(torch_data.Dataset):
    def __init__(self, root, list_name, mode, sample=50, Transform='simple', loader='cv', balanced=True):
        """
        FaceForensicsPP dataset
        :param root:
        :param mode:
        # :param compression:
        :param sample: image number per id.
        :param transform:
        :param loader:
        """
        self.root = root
        assert mode in ["train", "valid", "test"]
        self.mode = mode if mode != "valid" else "val"
        # self.compression = compression
        # assert self.compression in COMPRESSION
        self.num_sample = sample
        self.transform = Transform  #  TODO
        
        if Transform == 'simple':
            self.pixel_aug = Augmentator('simple')
            self.spatial_aug = None
        elif Transform == 'pixel':
            self.pixel_aug = Augmentator('pixel_aug')
            self.spatial_aug = None
        elif Transform == 'mild':
            self.pixel_aug = Augmentator('pixel_mild')
            self.spatial_aug = None
        elif Transform == 'mild_spatial':
            self.pixel_aug = Augmentator('pixel_mild')
            self.spatial_aug = Augmentator('spatial')
        else:
            raise NotImplementedError(Transform)

        self.trans_image = data_transform(normalize=True)
        self.trans_xray = data_transform(normalize=False)
        if self.mode != "train":
            self.num_sample = self.num_sample // 5
        if self.mode == 'valid':
            self.mode = 'val'


        assert loader in ['cv']
        self.loader = cv_loader
        self.balanced = balanced

        # FaceForensics
        self.pristine_id = []
        self.manipulated_id = []
        # read json file
        with open(os.path.join(root, 'splits', self.mode+".json"), 'r') as f:
            temp = json.loads(f.read())
            for pair in temp:
                self.pristine_id += pair
                self.manipulated_id += [pair[0], pair[1]]  # [pair[0]+'_'+pair[1]]
                # self.manipulated_id += [pair[1]+'_'+pair[0]]

        self.generate()

    def sample(self, image_names, decay_rate=1.0):
        assert 0 < decay_rate <= 1
        n_sample = int(decay_rate*self.num_sample)
        if self.num_sample == -1 or len(image_names) <= n_sample:
            return image_names
        return random.sample(image_names, n_sample)

    def generate(self):
        # FaceForensics
        # pristine_id to image_list

        self.files = []

        if self.mode == 'train':  methods = TRAIN_METHODS
        elif self.mode == 'val':  methods = VALID_METHODS
        elif self.mode == 'test': methods = TEST_METHODS
        else:
            raise NotImplementedError(self.mode)
        nums = self.generate_mode(methods)

        print("dataset size: {}, class nums: {}".format(len(self.files), nums))
        

    def add_files(self, method, id, cls_label, decay_rate=1.0):
        # print(method, id)
        id_path = os.path.join(self.root, DATASET_PATHS[method], IMAGE_PATH, id)
        img_pattern = os.path.join(id_path, '????.png')
        pattern_list = glob.glob(img_pattern)
        # if method !='original':
        #     import pdb; pdb.set_trace()
        for image_name in self.sample(pattern_list, decay_rate=decay_rate):
            image_path = os.path.join(id_path, image_name)
            pre, ext = os.path.splitext(image_path)
            if method in ORIGINAL_METHODS:
                xray_path = os.path.join(self.root, '0.png')
            else:
                xray_path = pre + '_label.png'
            self.files.append({
                "img": os.path.join(id_path, image_name),
                "label": xray_path,
                "cls_label": cls_label,
                "name": image_path
            })

    def generate_mode(self, methods):
        print(methods)
        decay_rate = 1.0
        if self.balanced:
            decay_rate = 1. / len(methods)

        for method in ORIGINAL_METHODS:
            for id in self.pristine_id:
                self.add_files(method, id, cls_label=0, decay_rate=1.0)
        num_neg = len(self.files)
                
        for method in methods:
            for id in self.manipulated_id:
                self.add_files(method, id, cls_label=1, decay_rate=decay_rate)
        num_pos = len(self.files) - num_neg

        return num_neg, num_pos

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        name = datafiles["name"]
        image = self.loader(datafiles["img"])
        if self.pixel_aug:
            image = self.pixel_aug(image)
        xray = self.loader(datafiles["label"], gray=True)
        if self.spatial_aug:
            image, xray = self.spatial_aug(image, xray)
        cls_label = int(datafiles["cls_label"])

        size_origin = image.size  # W * H

        # 将opencv image转换为Tensor
        image = self.trans_image(image=image)['image']
        xray = self.trans_xray(image=xray)['image']
        xray = xray.unsqueeze(0)

        return image, xray, cls_label, np.array(size_origin), name


if __name__ == '__main__':

    def run():
        root = "/mnt/hjr/FF++"

        for mode in ['train', 'val', 'test']:
            dataset = Xray_FFpp(root=root, mode=mode, sample=10, transform=None)
            trainloader = torch_data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False)
            print(len(dataset))
            for imgs, xrays, cls_labels, _, _ in trainloader:
                print(imgs.shape, imgs.min(), imgs.max())
                print(xrays.shape, xrays.min(), xrays.max())
                print(cls_labels)
                # print(data[0].shape, data[1])
                break
    run()