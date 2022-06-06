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
- Upload to Server (four GPUs)
- The project should be organized as:
```text
SOLC
├── dataset
|   |whu-opt-sar-dataset-256     //  root
│   ├──   ├── train
│   ├──   │     ├── sar
│   ├──   │     │     ├── NH49E001013_1.tif
│   ├──   │     ├── opt
│   ├──   │     │     ├── NH49E001013_1.tif
│   ├──   │     ├── lbl
│   ├──   │     │     ├── NH49E001013_1.tif
│   ├──   ├── val
│   ├──   │     ├── sar
│   ├──   │     │     ├── NH49E001013_2.tif
│   ├──   │     ├── opt
│   ├──   │     │     ├── NH49E001013_2.tif
│   ├──   │     ├── lbl
│   ├──   │     │     ├── NH49E001013_2.tif
│   ├──   ├── test
│   ├──   │     ├── sar
│   ├──   │     │     ├── NH49E001013_3.tif
│   ├──   │     ├── opt
│   ├──   │     │     ├── NH49E001013_3.tif
│   ├──   │     ├── lbl
│   ├──   │     │     ├── NH49E001013_3.tif
├── libs     //  utils
├── models     //  model
├── tools     //  preprocessing
├── dataset.py
├── class_names.py
├── palette.py 
├── sync_transforms.py 
├── train.py     
├── _test.py / predict.py
```
### 2022-06-02 News

- Release Deeplabv3+ Code (sar + opt, 5-channels input)
- Release Learning Strategy Code (step size, gamma)
- Release Training Torch-Parser Code
- Release VGG19 Code (based on official implement) [weights](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth)
 
### 2022-06-03 News
- Release MCANet Code (unofficial implement, based on deeplabv3+)
- Release FCNs Code (FCN8s, 16s and 32s) 
- Release Resnet-50 [weights](https://download.pytorch.org/models/resnet50-19c8e357.pth) , Resnet-101 [weights](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), and Restnet-152 [weights](https://download.pytorch.org/models/resnet152-b121ed2d.pth)  Code   
- Release Unet Code
- Release Segnet Code
- Release PSPnet Code
- Release SOLC Code (Ours) 


### 2022-06-04 News SOLC (Ours)
- Release SOLC V1 Code (based on RGB-D and dual-resnet 50) 
- Release SOLC V2 Code (based on dual-stream deeplabv3+)
- Release Deeplabv3+ Performance (Training 7 hours)
- Release Predict Code 
         
### 2022-06-05 News SOLC (Ours)
- Release SOLC V1 Code Performance
- Release SOLC V2 Code Performance
- Release SOLC V3 Code (based on dual-stream deeplabv3+ and SAGate)
- Release SOLC V4 Code (based on dual2one-stream deeplabv3+)
- Release SOLC V5 Code (based on dual2one-stream deeplabv3+, SAGate and RFB)
- Release SOLC V6 Code (based on dual2one-stream deeplabv3+ and Two Enhanced Module)


### 2022-06-06 News SOLC (Ours)
- Release SOLC V7 Code (based on dual-stream deeplabv3+, SAGate and ERFB) (successful version, **Congratulations!**)
- Retrain Unet, Segnet, and MCANet
- Retest the performance
- **Release Our SOLC V7 weights** solcv7: [baiduyun](https://pan.baidu.com/s/17DaI3e5alCWq2etOZDW5WQ)  password：solc
- Release Other model weights others: [baiduyun](https://pan.baidu.com/s/17DaI3e5alCWq2etOZDW5WQ)  password：solc


### Other stragety
- 设置合适的空洞卷积膨胀率atrous_rates
- 余弦退火重启动学习率策略warm up
- 使用更多的数据增强
- 使用更强的损失函数（focal loss）或者为类别赋予权重（见tools/class_weight.py） 来解决类别不平衡问题


### Performance

|                             策略                             |             模型              | 总体性能 |  
| :----------------------------------------------------------: | :---------------------------: | :--: | 
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | deeplabv3+ (pretrained=False) |  oa=0.8096,mIoU=0.4118,kappa=0.7261    |     
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | unet (pretrained=False)       |  oa=0.8207,mIoU=0.3649,kappa=0.7278    |                     
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | segnet (pretrained=False)     |  oa=0.8135,mIoU=0.3841,kappa=0.7159    |      
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | mcanet (pretrained=False)      |  oa=0.8388,mIoU=0.3958,kappa=0.7520    |     
| epoch=40, batch size=16, <br />Random Flip, lr=1e-3, wd=1e-4 | solcv7 (pretrained=False)      |  oa=0.8388,mIoU=0.3958,kappa=0.7520    |     



|    模型    | farmland | city | village | water | forest | road | others | background(ignored) |
| :--------: | :------: | :--: | :-----: | :---: | :----: | :--: | :----: | :-----------------: |
| deeplabv3+ |     /     |   /   |     /    |    /   |   /     |   /   |    /    |       /              |
|    unet    |    /      |   /   |    /     |    /   |   /     |   /   |    /    |       /              |
|   segnet   |     /     |   /   |    /     |    /   |    /    |   /   |    /    |       /              |
|   mcanet   |    /      |   /   |    /     |    /   |    /    |   /   |    /    |      /               |
|   solcv7   |    /      |   /   |    /     |    /   |    /    |   /   |    /    |       /              |

### Installation
1. Clone this repo.
```shell
$ git clone https://github.com/yisun98/SOLC.git
$ cd SOLC
```
2. Install Environments

   ```shell
   $ pip install -r requirements.txt
   $ source activate
   ```
3. Dataset

   ```shell
   $ python tools/crop_sar.py
   $ python tools/crop_opt.py
   $ python tools/convert_lbl.py
   $ python tools/crop_lbl.py
   $ python tools/split_data.py
   ```
4. Training

```shell
nohup python train.py >> train_<model_name>.out 2>&1 &
```
Please see train.py for details.

```shell
tensorboard --logdir=<your_log_dir> --bind_all 
```

### Test/Predict
```shell
nohup python train.py --model solcv7 --num_classes 8 >> train_<model_name>.out 2>&1 &
```
Please see train.py for details.

```shell
python _test.py or python predict.py --model solcv7 --model-path <model_path>
```



