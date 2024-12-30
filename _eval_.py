import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import TestDataset, make_train_data, ClassificationDataset, SupportDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle

import time
from tqdm import tqdm

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def evaluation(encoder, bn, decoder, dataloader, device, shot, _class_=None):
    #Hyper params
    image_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = '../mvtec/' + _class_
    # support data
    fewshot_data = SupportDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, shot=shot)
    fewshot_dataloader = torch.utils.data.DataLoader(fewshot_data, batch_size=1,  num_workers=1, shuffle=True)
    
    # 評価モード
    encoder.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    
    auc_list = []
    time_list = []

    for i in tqdm(range(10)):
        # # support img
        # support_img = augment_support_data(fewshot_dataloader)

        # start_time = time.perf_counter()
        # with torch.no_grad():
        #     support_img = support_img.to(device)
        #     inputs = encoder(support_img)
        #     btl, z_support, x = bn(inputs)
            
            # outputs = decoder(btl)
        start_time = time.perf_counter()
        with torch.no_grad():
            img_num = 0
            for img, gt, label, _ in dataloader:

                img = img.to(device)
                inputs = encoder(img)
                _btl, z_test, _x = bn(inputs)
                outputs = decoder(_btl)

                # outputs = decoder(bn(inputs))
                
                # segmentation
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                # detection
                # anomaly_vector = cal_score(btl, _btl)

                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                # if label.item()!=0:
                #     aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                #                                   anomaly_map[np.newaxis,:,:]))
                #gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                #pr_list_px.extend(anomaly_map.ravel())

                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))
                img_num += 1 
            #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
            #vis_data = {}
            #vis_data['Anomaly Score'] = ano_score
            #vis_data['Ground Truth'] = np.array(gt_list_sp)
            # print(type(vis_data))
            # np.save('vis.npy',vis_data)
            #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
            #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


        #auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        auc_list.append(auroc_sp)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time / img_num)
    
    #return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

    return sum(auc_list) / len(auc_list), sum(time_list) / len(time_list)