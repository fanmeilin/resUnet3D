import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


class PetDataset(Dataset):
    def __init__(self, noise_dir, clean_dir):
        super(PetDataset, self).__init__()
        self.pet_data = []
        self.pet_label = []

        # self.noise_dir = self.base_dir + '/LD_pos'  # 泊松噪声
        self.noise_dir = noise_dir  # 脑PET噪声
        self.clean_dir = clean_dir

        # 噪声数据
        for image in os.listdir(self.noise_dir):
            image_dir = self.noise_dir / image
            image_dir = str(image_dir)
            data = self.read_img(image_dir)
            # data = np.array(data, 'float32')
            data = (data - data.min()) / (data.max() - data.min())  # 归一化
            data -= data.mean()  # 标准化
            data /= data.std()
            self.pet_data.append(data)
        # 干净数据
        for image in os.listdir(self.clean_dir):
            image_dir = self.clean_dir / image
            image_dir = str(image_dir)
            data = self.read_img(image_dir)
            # data = np.array(data, 'float32')
            data = (data - data.min()) / (data.max() - data.min())
            data -= data.mean()
            data /= data.std()
            self.pet_label.append(data)

        # 随机打乱
        index = [i for i in range(len(self.pet_label))]
        random.shuffle(index)
        self.pet_data = np.array(self.pet_data)
        self.pet_label = np.array(self.pet_label)
        self.pet_data = self.pet_data[index]
        self.pet_label = self.pet_label[index]

    def read_img(self, path):
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        data = data[np.newaxis, :, :, :]
        return data

    def __getitem__(self, index):
        images = self.pet_data[index]
        label = self.pet_label[index]
        images = torch.Tensor(images).float() #增加.float()
        return images, label

    def __len__(self):
        return len(self.pet_label)
