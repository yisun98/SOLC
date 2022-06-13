import os, glob
from PIL import  Image
import numpy as np

from collections import Counter

root = r'E:\Segmentation\whu-opt-sar\lbl'
files = os.listdir(root) # 1-7

for file in files[0:11]:
    data = np.array(Image.open(os.path.join(root, file)))
    unique, count = np.unique(data, return_counts=True)
    data_count = dict(zip(unique, count))
    print(data_count)

# if __name__ == '__main__':
#     x = [1,2,3,4,5][2:4]
#     print(x)