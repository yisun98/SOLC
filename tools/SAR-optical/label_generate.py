import numpy as np
import glob, os, h5py
import os
import cv2
from scipy import misc
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

lbl_new_map = {
    0: 1,
    10: 2,
    20:3,
    30:4,
    40:5,
    50:6,
    60:7,
    70:8
}

def old_label2new_label(lbl, new_map, root_path, name):
    h, w = lbl.shape
    # newlbl = np.zeros((h, w))

    lbl[np.where(lbl == 0)] = 1
    lbl[np.where(lbl == 10)] = 2
    lbl[np.where(lbl == 20)] = 3
    lbl[np.where(lbl == 30)] = 4
    lbl[np.where(lbl== 40)] = 5
    lbl[np.where(lbl == 50)] = 6
    lbl[np.where(lbl == 60)] = 7
    lbl[np.where(lbl == 70)] = 8

    # for i in range(h):
    #     for j in range(w):
    #         # print(classmap[i, j], '->', palette[classmap[i, j], :])
    #         newlbl[i, j] = new_map[lbl[i, j]]

    path = os.path.join(root_path, name)
    print('=====> save', path)
    Image.fromarray(np.uint8(lbl[:, :])).convert('L').save(path)

    return

if __name__ == '__main__':

    # data = np.array(Image.open('E:\Segmentation\whu-opt-sar\lbl\\NH49E001013.tif'))
    # print(data.shape, data.max(), data.min())
    # lbl_root = r'E:\Segmentation\whu-opt-sar\lbl'
    # new_lbl_root = r'E:\Segmentation\whu-opt-sar\newlbl'
    # label_files = sorted(os.listdir(lbl_root))
    # for file_name in label_files:
    #     data = np.array(Image.open(os.path.join(lbl_root, file_name)))
    #     old_label2new_label(data, lbl_new_map, new_lbl_root, file_name)

    data1 = np.array(Image.open('E:\Segmentation\whu-opt-sar\lbl\\NI49E023017.tif'))
    print(data1, data1.max(), data1.min())

    data2 = np.array(Image.open('E:\Segmentation\whu-opt-sar\\newlbl\\NI49E023017.tif'))
    print(data2, data2.max(), data2.min())


