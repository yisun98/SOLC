# !/usr/bin/env python
# coding=utf-8

#### crop size 256 x 256, stride 256
import numpy as np
import glob, os, h5py
import os
import cv2
from scipy import misc
from PIL import Image

Image.MAX_IMAGE_PIXELS = 999999999999

class image_to_patch:
    def __init__(self, patch_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path):
        self.stride = patch_size

        self.sar_path = sar_path
        self.optical_path = optical_path

        self.crop_sar_image_path = crop_sar_image_path
        self.crop_optical_image_path = crop_optical_image_path

        print(self.crop_sar_image_path, self.crop_optical_image_path)

        if not os.path.exists(crop_sar_image_path):
            os.mkdir(crop_sar_image_path)

        if not os.path.exists(crop_optical_image_path):
            os.mkdir(crop_optical_image_path)

    def to_patch(self):

        n_sar = 1
        n_optical = 1

        # sar:  NH49E001013.tif
        # optical: NH49E001013.tif

        sar_files = sorted(os.listdir(self.sar_path))
        optical_files = sorted(os.listdir(self.optical_path))

        print(len(sar_files), len(optical_files))
        # assert len(sar_files) == len(optical_files)

        for file_name in sar_files:
            prefix = file_name.split('.')[0]
            sar_path = os.path.join(self.sar_path, file_name)

            optical_path = os.path.join(self.optical_path, file_name)

            # read Optical image
            # img_optical = self.imread(optical_path)
            fp2 = open(optical_path, 'rb')
            img_optical = Image.open(fp2)
            # high, width => equal size
            h, w = img_optical.size

            # OPTICAL image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    box = [x, y, x + self.stride, y + self.stride]
                    sub_img_label = img_optical.crop(box)
                    print('====> save', os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'))
                    sub_img_label.save(os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'))
                    sub_img_label.close()
                    n_optical = n_optical + 1
            fp2.close()
            img_optical.close()

    def imread(self, path):
        img = Image.open(path)
        return img


if __name__ == '__main__':
    image_size = 512

    sar_path = r'E:\Segmentation\whu-opt-sar\sar'
    crop_sar_image_path = r'E:/Segmentation/whu-opt-sar/crop/sar'

    optical_path = r'E:\Segmentation\whu-opt-sar\optical'
    crop_optical_image_path = r'E:/Segmentation/whu-opt-sar/crop/optical'

    # image to patch
    task = image_to_patch(image_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path)
    task.to_patch()