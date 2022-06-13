import gdal
import os
import glob
import numpy as np
import random
from tqdm import tqdm
from shutil import copy
import pandas as pd
import csv

input_data_path = r'E:\Segmentation\whu-opt-sar\wos10'


test_path = os.path.join(input_data_path, 'test')
test_sar_path = glob.glob(os.path.join(os.path.join(test_path, 'sar'), '*.tif'))
test_opt_path = glob.glob(os.path.join(os.path.join(test_path, 'optical'), '*.tif'))
test_lbl_path = glob.glob(os.path.join(os.path.join(test_path, 'lbl'), '*.tif'))

train_path = os.path.join(input_data_path, 'train')
train_sar_path = glob.glob(os.path.join(os.path.join(train_path, 'sar'), '*.tif'))
train_opt_path = glob.glob(os.path.join(os.path.join(train_path, 'optical'), '*.tif'))
train_lbl_path = glob.glob(os.path.join(os.path.join(train_path, 'lbl'), '*.tif'))




# train
print('开始写入数据集')
print("train set")

f = open(os.path.join(input_data_path, 'whu10-train.csv'), 'w', encoding='utf-8', newline="")

csv_write = csv.writer(f)

for i in tqdm(range(0, len(train_sar_path))):

    sar_path = os.path.join(train_sar_path[i]).replace('\\', '/')
    opt_path = os.path.join(train_opt_path[i]).replace('\\', '/')
    lbl_path = os.path.join(train_lbl_path[i]).replace('\\', '/')

    csv_write.writerow([sar_path, opt_path, lbl_path])





# test
print("test set")
f = open(os.path.join(input_data_path, 'whu10-test.csv'), 'w', encoding='utf-8', newline="")

csv_write = csv.writer(f)
for i in tqdm(range(0, len(test_sar_path))):

    sar_path = os.path.join(test_sar_path[i]).replace('\\', '/')
    opt_path = os.path.join(test_opt_path[i]).replace('\\', '/')
    lbl_path = os.path.join(test_lbl_path[i]).replace('\\', '/')

    csv_write.writerow([sar_path, opt_path, lbl_path])





