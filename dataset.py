from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

# Few-shot query Dataset
class FewshotDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type
    
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            # good : 0をリストに追加
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            # anomaly : 0をリストに追加
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

# Few-shot support Dataset
class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, shot):

        # add train path
        self.img_path = os.path.join(root, 'train')

        self.transform = transform
        self.gt_transform = gt_transform
        self.shot = shot

        # load dataset
        self.img_paths= self.load_dataset()

    def __getitem__(self, idx):
        img_path, gt = self.img_paths[idx], self.gt_paths[idx]

        support_img = []
        support_gt = []

        for k in range(self.shot):
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            support_img.append(img)
            support_gt.append(gt)

            assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return support_img, support_gt
    
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []

        img_paths = glob.glob(os.path.join(self.img_path) + "/*.png")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.img_paths)

# train
def make_train_data(exclude_label):
    # データセットのパスを設定
    output_path = '../mvtec_train'

    # ラベル情報を読み込む
    labels_df = pd.read_csv(os.path.join(output_path, 'labels.csv'))

    # 特定のラベルを除外
    filtered_df = labels_df[labels_df['label'] != exclude_label]

    # ラベルをエンコード
    label_encoder = LabelEncoder()
    filtered_df.loc[:, 'label'] = label_encoder.fit_transform(filtered_df['label'])
    # 訓練セットとテストセットに分割
    return filtered_df

# 分類タスク用
class ClassificationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(224, Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)
            
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label