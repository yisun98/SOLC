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
### 2022-06-02 News

- Release Deeplabv3+ Code (sar + opt, 5-channels)
- Release Learning Strategy Code (step size, gamma)
- Release Torch-Parser Code
- Release VGG19 Code (https://download.pytorch.org/models/vgg19_bn-c79401a0.pth)

```text
nohup python train.py >> train_deeplabv3.out 2>&1 &
```

|                             策略                             |             模型              | 总体性能 |  各类别    |
| :----------------------------------------------------------: | :---------------------------: | :--: | ---- |
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | deeplabv3+ (pretrained=False) |  oa=0.8096,mIoU=0.4118,kappa=0.7261    |      |
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | unet (pretrained=False) |     |  oa=0.8207,mIoU=0.3649,kappa=0.7278    |      |                
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | segnet (pretrained=False) |   |  oa=0.8135,mIoU=0.3841,kappa=0.7159    |      |   
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | mcanet                    |   |  oa=0.8388,mIoU=0.3958,kappa=0.7520    |      |        

### 2022-06-03 News
- Release MCANet Code (Reimplement, based on deeplabv3+)
- Release FCNs Code 
- Release Resnet-50, Resnet-152 Code 
- Release SOLC Code (Ours) 
```text
tensorboard 
```

### 2022-06-04 News SOLC (Ours)
- Release SOLC Code (based on RGB-D) : oa=0.7779,mIoU=0.4275,kappa=0.7047
- Release SOLC V2 Code (based on deeplabv3+): 
- Release Deeplabv3+ Performance
- Release Predict Code
         
 
### 2022-06-05 News SOLC (Ours)
- Release SOLC Code (based on RGB-D) : oa=0.7779,mIoU=0.4275,kappa=0.7047
- Release SOLC V2 Code (based on deeplabv3+): oa= 0.7994,mIoU=0.4832,kappa=0.7363





