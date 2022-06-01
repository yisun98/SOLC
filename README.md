# SOLC
Remote Sensing Sar-Optical Land-use Classfication Pytorch 

### Source Dataset

Refer to https://github.com/AmberHen/WHU-OPT-SAR-dataset.

Datasets：Sar and Optical

https://pan.baidu.com/s/1sIGsD3lBEogSCqzbDOaclA password：i51o

Paper Link: MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification

https://www.sciencedirect.com/science/article/pii/S0303243421003457

### 2022-06-01 News
- Release Crop Code (sar, opt, lbl)
- Release Convert Label Code 
- Release Split Code (6:2:2) -> (17640:5880:5880)
- Upload
- The project should be organized as:
```text
SOLC
├── whu-opt-sar-dataset-256     //  root
│   ├── train
│   │     ├── sar
│   │     │     ├── NH49E001013_1.tif
│   │     ├── opt
│   │     │     ├── NH49E001013_1.tif
│   │     ├── lbl
│   │     │     ├── NH49E001013_1.tif
│   ├── val
│   │     ├── sar
│   │     │     ├── NH49E001013_2.tif
│   │     ├── opt
│   │     │     ├── NH49E001013_2.tif
│   │     ├── lbl
│   │     │     ├── NH49E001013_2.tif
│   ├── test
│   │     ├── sar
│   │     │     ├── NH49E001013_3.tif
│   │     ├── opt
│   │     │     ├── NH49E001013_3.tif
│   │     ├── lbl
│   │     │     ├── NH49E001013_3.tif
```
