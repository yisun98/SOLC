# !/usr/bin/env python
# coding=utf-8

#### crop size 256 x 256, stride 256
import numpy as np
import glob, os, h5py
import os
import cv2
from scipy import misc
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class image_to_patch:
    def __init__(self, patch_size, lbl_path, crop_lbl_image_path, left, right):
        self.stride = patch_size

        self.lbl_path = lbl_path
        self.crop_lbl_image_path = crop_lbl_image_path


        self.left = left
        self.right = right

        if not os.path.exists(crop_lbl_image_path):
            os.mkdir(crop_lbl_image_path)


    def to_patch(self):



        # lbl:  NH49E001013.tif

        lbl_files = sorted(os.listdir(self.lbl_path))

        print(len(lbl_files))
        n_lbl = 1

        for file_name in lbl_files[self.left:self.right]:

            prefix = file_name.split('.')[0]
            lbl_path = os.path.join(self.lbl_path, file_name)

            # read lbl image
            img_lbl = Image.open(lbl_path)

            # high, width => equal size
            h, w = img_lbl.size

            # SAR image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    box = [x, y, x + self.stride, y + self.stride]
                    sub_img_label = img_lbl.crop(box)
                    print('====> sar save', os.path.join(self.crop_lbl_image_path,  prefix + '_' + str(n_lbl) + '.tif'))
                    sub_img_label.save(os.path.join(self.crop_lbl_image_path,  prefix + '_' + str(n_lbl) + '.tif'))
                    n_lbl = n_lbl + 1
            img_lbl.close()



    def imread(self, path):
        img = Image.open(path)
        return img


if __name__ == '__main__':
    image_size = 256

    lbl_path = r'E:\Segmentation\whu-opt-sar\lbl'
    crop_lbl_image_path = r'E:/Segmentation/whu-opt-sar/wos10/lbl'

    # image to patch
    task = image_to_patch(image_size, lbl_path, crop_lbl_image_path, 0, 11) # top 10 labels
    task.to_patch()