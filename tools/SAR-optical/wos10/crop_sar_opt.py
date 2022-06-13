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
    def __init__(self, patch_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path, left, right):
        self.stride = patch_size

        self.sar_path = sar_path
        self.optical_path = optical_path

        self.crop_sar_image_path = crop_sar_image_path
        self.crop_optical_image_path = crop_optical_image_path


        self.left = left
        self.right = right

        if not os.path.exists(crop_sar_image_path):
            os.mkdir(crop_sar_image_path)

        if not os.path.exists(crop_optical_image_path):
            os.mkdir(crop_optical_image_path)

    def to_patch(self):



        # sar:  NH49E001013.tif
        # optical: NH49E001013.tif

        sar_files = sorted(os.listdir(self.sar_path))
        optical_files = sorted(os.listdir(self.optical_path))

        print(len(sar_files), len(optical_files))
        # assert len(sar_files) == len(optical_files)

        n_sar = 1
        n_optical = 1

        for file_name in sar_files[self.left:self.right]:

            prefix = file_name.split('.')[0]
            sar_path = os.path.join(self.sar_path, file_name)
            optical_path = os.path.join(self.optical_path, file_name)

            # read SAR image
            # img_sar = self.imread(sar_path)
            img_sar = Image.open(sar_path)

            # high, width => equal size
            h, w = img_sar.size

            # SAR image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    box = [x, y, x + self.stride, y + self.stride]
                    sub_img_label = img_sar.crop(box)
                    print('====> sar save', os.path.join(self.crop_sar_image_path,  prefix + '_' + str(n_sar) + '.tif'))
                    sub_img_label.save(os.path.join(self.crop_sar_image_path,  prefix + '_' + str(n_sar) + '.tif'))
                    n_sar = n_sar + 1
            img_sar.close()

            # read Optical image
            img_optical = self.imread(optical_path)
            # high, width => equal size
            h, w = img_optical.size

            # OPTICAL image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    box = [x, y, x + self.stride, y + self.stride]
                    sub_img_label = img_optical.crop(box)
                    print('====> optical save', os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'))
                    sub_img_label.save(os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'))
                    n_optical = n_optical + 1
            img_optical.close()



    def imread(self, path):
        img = Image.open(path)
        return img


if __name__ == '__main__':
    image_size = 256

    sar_path = r'E:\Segmentation\whu-opt-sar\sar'
    crop_sar_image_path = r'E:\Segmentation\whu-opt-sar\wos10/sar'

    optical_path = r'E:\Segmentation\whu-opt-sar\optical'
    crop_optical_image_path = r'E:\Segmentation\whu-opt-sar\wos10/optical'

    # image to patch
    task = image_to_patch(image_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path, 0, 11) # top 10 images
    task.to_patch()