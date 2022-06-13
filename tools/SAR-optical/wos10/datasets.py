import os
import cv2
import csv
import ast
import json
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from keras.utils.np_utils import *
import pandas


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.len = SegDataset._read_image_ids(self.is_train)
        print('dataset size ====>', self.len)

    def __len__(self):
        return len(self.len)

    @staticmethod
    def _read_image_ids(is_train):

        if is_train:
            data = pd.read_csv('/home/sy/whu-opt-sar/whu10-train.csv').values
            random.seed(1)
            random.shuffle(data)
            return data

        else:
            data = pd.read_csv('/home/sy/whu-opt-sar/whu10-train.csv').values
            random.seed(1)
            random.shuffle(data)
            return data

    def __getitem__(self, idx):
        train_sar_path, train_opt_path, label_path = self.len[idx][0], self.len[idx][1], self.len[idx][2]
        label_path = label_path.strip('\n')
        train_sar_path = '' + train_sar_path
        train_opt_path = '' + train_opt_path
        label_path = '' + label_path

        # train = cv2.imread(train_path)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        train_sar = Image.open(train_sar_path)
        train_opt = Image.open(train_opt_path)
        label = Image.open(label_path, mode='L')

        label.flags.writeable = True
        train_sar = np.asarray(train_sar)
        train_opt = np.asarray(train_opt)
        label = np.asarray(label)
        label.flags.writeable = True
        label_seg = label

        train_sar = np.transpose(train_sar, (2, 0, 1))
        train_sar = train_sar.astype(np.float32)
        imgA = torch.from_numpy(train_sar)

        train_opt = np.transpose(train_opt, (2, 0, 1))
        train_opt = train_opt.astype(np.float32)
        imgB = torch.from_numpy(train_opt)

        imgC = torch.FloatTensor(label_seg).long()
        # print(torch.min(imgB), torch.max(imgB))  question

        item = {'A': imgA, 'B': imgB, 'C': imgC}
        return item