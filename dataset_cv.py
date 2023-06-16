import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from strong import gamma
from PIL import Image
from  get_LBP_from_Image import LBP
from matplotlib import pyplot as plt
from tools.msdb import multiScaleSharpen



class Mydata(Dataset):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        file_list_label = os.listdir(self.root)  # ['1', '100']

        self.label = []
        self.data = []
        lbp = LBP()



        for index, item in enumerate(file_list_label):  # index:0,1;    # i:'1', '100'
            file_list_img = os.listdir(self.root + '/' + item + '/')  # './RMB_data/1/', './RMB_data/100/'
            for j in file_list_img:  # ['1-1.jpg', .......]

                imge = cv.imread(root + '/' + item + '/' + j)
                imge = multiScaleSharpen(imge, 5)

                #检查是否为三通道
                # try:
                #     r, g, b = imge.split()
                # except Exception:
                #     print("no rgb！")


                #yBRCR
                Ycrcb = cv.cvtColor(imge, cv.COLOR_BGR2YCrCb)  # 转化 Ycrcb 格式的图像
                b, g, r = cv.split(Ycrcb)


                self.label.append(index)
                self.data.append(b)



    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            imgee = self.transform(img)
            # print(imgee.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgee, target

    def __len__(self):
        return len(self.data)

