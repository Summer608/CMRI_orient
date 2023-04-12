import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import glob
from torchvision import transforms
from PIL import Image
import cv2

class dataloader(Dataset):
    def __init__(self, root=None, mode='train', truncation=False):
        '''

        :param root: 根目录
        :param mode: train valid test
        :param truncation: 是否截断concat
        '''
        super(dataloader, self).__init__()
        self.root = root
        self.mode = mode
        self.truncation = truncation
        print(f"load data in: {self.root} as {self.mode}")

        #数据集train/valid/test
        self.data = []
        if self.mode == 'train':
            for i in range(1, 32):
                self.data += glob.glob(self.root+f'/*/patient{i}*')
        elif self.mode == 'valid':
            for i in range(32, 46):
                self.data += glob.glob(self.root+f'/*/patient{i}*')
        elif self.mode == 'test':
            self.data += [self.root]
        print(self.data)
        print(f'len of data = {len(self.data)}')

        #数据增强
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(), #张量转换为PIL图像
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'valid': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                #transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
                #transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

    def __len__(self):
        return len(self.data)

    #获取数据集
    def __getitem__(self, item):
        img_fname = self.data[item]
        img = cv2.imread(img_fname, 0) #获取灰度图 H*W

        if self.truncation:
            #截断扩展到3通道
            img100, img80, img60 = img.copy(), img.copy(), img.copy
            ret1, img60 = cv2.threshold(img, 0.6*np.max(img), 0.6*np.max(img), cv2.THRESH_TRUNC)#60%阈值截断
            ret1, img80 = cv2.threshold(img, 0.8 * np.max(img), 0.8 * np.max(img), cv2.THRESH_TRUNC)

            #直方图均衡
            img60 = cv2.equalizeHist(img60)
            img80 = cv2.equalizeHist(img80)
            img100 = cv2.equalizeHist(img100)

            img = cv2.merge([img60, img80, img100])

            #维度变化 H*W*T --> T*H*W
            img = torch.tensor(img).float()
            img = img.permute(2, 0, 1)
        else:
            img = torch.tensor(img).float()

        if self.mode != 'test':
            label = int(img_fname.split('\\')[-2]) #以‘/ ’为分割符，保留倒数第二个
            label = torch.tensor(label).long()
            img = self.data_transforms[self.mode](img)
            return img, label
        else:
            img = self.data_transforms[self.mode](img)
            return img




# if __name__=='__main__':
#
#
#     root = 'D:/PycharmProjects/pj_orient/data/data0/C0'
#
#
#
#     dataset = dataloader(root=root, mode='valid', truncation = True)
#     load_dataset = DataLoader(dataset, batch_size=32, shuffle=True,
#                                 num_workers=0)
#     for i, (i1, i2) in enumerate(load_dataset):
#         print(i1.shape)
#         print(i2)
#         break
