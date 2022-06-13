import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from models.deeplabv3_version_1.deeplabv3 import DeepLabV3
from torch.autograd import Variable
import torch
import os
import pandas as pd
from PIL import Image
import cv2 as cv
from collections import OrderedDict
import torch.nn as nn
from dataset import WHUOPTSARDataset
from dataset import img_sar_transform, img_opt_transform, mask_transform
from models.deeplabv3_version_3.deeplabv3 import DeepLabV3 as SOSeg
from palette import colorize_mask
from torchvision import transforms
from libs import average_meter, metric
from models.SOLC.solc import SOLC
from models.SOLCV2.solcv2 import SOLCV2
from models.SOLCV5.solcv5 import SOLCV5
from models.SOLCV7.solcv7 import SOLCV7
from models.MCANet.mcanet import MCANet
from models.u_net import UNet
from models.seg_net import SegNet

img_transform = transforms.Compose([
    transforms.ToTensor()])
    
resore_transform = transforms.Compose([
    transforms.ToPILImage()
])

from class_names import eight_classes
class_name = eight_classes()


def snapshot_forward(model, dataloader, save_path, num_classes):
    model.eval()
    for index, data in enumerate(dataloader):
        imgs_sar = Variable(data[0])
        imgs_opt = Variable(data[1])
        masks = Variable(data[2])
        # print(imgs_sar.shape, imgs_opt.shape, masks.shape)

        imgs_sar = imgs_sar.cuda()
        imgs_opt = imgs_opt.cuda()
        masks = masks.cuda()


        outputs = model(imgs_sar, imgs_opt)
        preds = torch.argmax(outputs, 1)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
        conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
        
        for i in range(masks.shape[0]):
            
            img_pil = resore_transform(imgs_opt[i])
            img_sar = resore_transform(imgs_sar[i])
            preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')
            pred_vis_pil = colorize_mask(preds[i])
            gt_vis_pil = colorize_mask(masks[i])
            data = np.array(img_pil)[:, :, :3]
            img_pil = Image.fromarray(np.uint8(data[:, :]))
            img_sar = Image.fromarray(np.uint8(img_sar))

            dir_list = ['opt', 'label', 'sar', 'predict', 'gt']
            rgb_save_path = os.path.join(save_path, dir_list[0] )
            sar_save_path = os.path.join(save_path, dir_list[2] )
            label_save_path = os.path.join(save_path, dir_list[1])
            vis_save_path = os.path.join(save_path, dir_list[3])
            gt_save_path = os.path.join(save_path, dir_list[4])

            path_list = [rgb_save_path, label_save_path, sar_save_path, vis_save_path, gt_save_path]
            for path in range(5):
                if not os.path.exists(path_list[path]):
                    os.makedirs(path_list[path])
            img_pil.save(os.path.join(path_list[0], 'img_opt_batch_%d_%d.jpg' % (index, i)))
            img_sar.save(os.path.join(path_list[2], 'img_sar_batch_%d_%d.jpg' % (index, i)))
            preds_pil.save(os.path.join(path_list[1], 'label_%d_%d.png' % (index, i)))
            pred_vis_pil.save(os.path.join(path_list[3], 'pre_%d_%d.png' % (index, i)))
            gt_vis_pil.save(os.path.join(path_list[4], 'gt_%d_%d.png' % (index, i)))
            
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                    label=masks.flatten(),
                                                    num_classes=num_classes)
    test_acc, test_acc_per_class, test_acc_cls, test_IoU, test_mean_IoU, test_kappa = metric.evaluate(conf_mat)
    print("test_acc:", test_acc)
    print("test_mean_IoU:", test_mean_IoU)
    print("test kappa:", test_kappa)
    for i in range(num_classes):
        print(i, eight_classes()[i], test_acc_per_class[i], test_IoU[i])        


def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--test-data-root', type=str, default=r'/data/sy/whu-opt-sar-dataset-256/test')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--model-path", type=str,
                        default=r"/data/sy/experiments-whu-opt-sar-dataset-256/mcanet/06-06-11:39:37/epoch_37_oa_0.79745_kappa_0.71153_latest.pth")
    parser.add_argument("--pred-path", type=str, default="/data/sy/download-SOLC/mcanet-best")
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='') # len=16
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    args = parser.parse_args()
    return args




def reference():
    args = parse_args()
    n_blocks = args.n_blocks
    n_blocks = [int(b) for b in n_blocks.split(',')]
    atrous_rates = args.deeplabv3_atrous_rates
    atrous_rates = [int(s) for s in atrous_rates.split(',')]
    multi_grids = args.multi_grids
    multi_grids = [int(g) for g in multi_grids.split(',')]
    dataset = WHUOPTSARDataset(class_name=class_name,
                               root=args.test_data_root,
                               img_sar_transform=img_sar_transform, img_opt_transform=img_opt_transform, mask_transform=mask_transform,
                               sync_transforms=None
                              )
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=8)
    print(class_name, len(class_name))
    """
    model = SOSeg(num_classes=len(class_name),
                          n_blocks=n_blocks,
                          atrous_rates=atrous_rates,
                          multi_grids=multi_grids,
                          output_stride=args.output_stride)
    """
    
    model = MCANet(num_classes=len(class_name))
    # model = MCANet(num_classes=len(class_name))
    
    # model = SOSeg(num_classes=len(class_name),
     #                     n_blocks=n_blocks,
     ##                     atrous_rates=atrous_rates,
     #                     multi_grids=multi_grids,
      #                    output_stride=args.output_stride)
                           
    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('=========> load model success', args.model_path)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])

    snapshot_forward(model, dataloader, args.pred_path, len(class_name))
    print('test done........')
if __name__ == '__main__':
    reference()