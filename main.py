# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset, make_train_data, ClassificationDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F
from loss import center_loss_func, update_center, loss_fucntion, loss_concat
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(_class_, class_list):

    for i in range(len(class_list)):
        # 対象カテゴリ以外を省く
        if class_list[i] == _class_:
           class_list.pop(i)
           break

    print(_class_)
    print(class_list)

    # Hyper params:
    epochs = 5
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    num_class = len(class_list)
    center_alpha = 0.5
    center_beta = 0.5
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # train dataframe
    train_df = make_train_data(_class_)

    # dataset
    train_dataset = ClassificationDataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    # train_path = '../mvtec/' + _class_ + '/train'
    test_path = '../mvtec/' + _class_
    ckp_path = '../checkpoints/' + 'wres50_'+_class_+'.pth'
    #train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    #train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # encoder and bottleneck: learnable
    encoder, bn = wide_resnet50_2(num_class, pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    # encoder.train()

    # decoder: freeze
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    # decoder.eval()

    # print model
    # print(bn)

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    
    start_time = time.perf_counter()
    for epoch in range(epochs):
        encoder.train()
        bn.train()
        decoder.train()
        loss_list = []

        # watching loss
        cosloss_list = []
        centerloss_list = []

        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device, dtype=torch.int64)
            inputs = encoder(img)
            btl, z, x = bn(inputs)
            outputs = decoder(btl)

            # 損失計算
            cosloss = loss_fucntion(inputs, outputs)
            centerloss = F.cross_entropy(x, label)  + center_alpha * center_loss_func(bn, z, label)
            loss = cosloss + centerloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_center(bn, label, center_beta, num_class)
            cosloss_list.append(cosloss.item())
            centerloss_list.append(centerloss.item())
            loss_list.append(loss.item())
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print('cos_loss: {:.4f}, center_loss: {:.4f}'.format(np.mean(cosloss_list), np.mean(centerloss_list)))
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        print("epoch {}, time:{} m {} s".format(epoch + 1, int(elapsed_time // 60), int(elapsed_time % 60)))
        if (epoch + 1) % 10 == 0:
            # auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    # return auroc_px, auroc_sp, aupro_px

if __name__ == '__main__':

    setup_seed(111)
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    
    for i in item_list:
        train(i, item_list)