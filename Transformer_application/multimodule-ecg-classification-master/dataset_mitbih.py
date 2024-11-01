# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/10/22 10:27
@Author  : Rao Zhi
@File    : dataset_mitbih.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils.data as data


class Ecg_loader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super(Ecg_loader, self).__init__()
        self.male_vec = pd.read_csv(os.path.join(path, 'res', 'male.csv'), header=None).to_numpy()[:, 0]
        self.female_vec = pd.read_csv(os.path.join(path, 'res', 'female.csv'), header=None).to_numpy()[:, 0]

        with open(os.path.join(path, 'ecg_labels.json')) as j_file:
            json_data = json.load(j_file)
        self.idx2name = json_data['labels']
        data = json_data['data']
        self.inputs = []
        self.labels = []
        self.gender = []
        self.inputs_full = []
        self.whole_ecg = []
        self.ecg = []
        self.age = []
        for i in tqdm(data):
            subject_img = []
            subject_ecg = []
            a = np.zeros((100))
            for i_name, w_name in zip(i['images'], i['ecg']):
                img = cv2.imread(os.path.join(path, 'images', transform, i_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (90, 90))
                ecg = np.load(os.path.join(path, 'ecg', w_name))
                subject_img.append(np.expand_dims(img.transpose((2, 0, 1)), axis=0))
                subject_ecg.append(np.expand_dims(np.expand_dims(ecg, axis=0), axis=0))
            img_full = cv2.imread(os.path.join(path, 'images_full', transform, i['images_full']))
            img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
            l = i['label']
            a[int(i['age'] * 100)] = 1
            if i['gender'] == [0, 1]:
                g = self.male_vec
            elif i['gender'] == [1, 0]:
                g = self.female_vec
            self.inputs_full.append(img_full.transpose((2, 0, 1)))
            self.inputs.append(np.concatenate(subject_img, axis=0))
            self.ecg.append(np.concatenate(subject_ecg, axis=0))
            self.whole_ecg.append(np.concatenate(subject_ecg, axis=2))
            self.labels.append(np.array(l))
            self.gender.append(g)
            self.age.append(a)
        print(len(self.whole_ecg))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        a = torch.from_numpy(np.array(self.age[idx])).float()
        g = torch.from_numpy(np.array(self.gender[idx])).float()
        w = torch.from_numpy(self.ecg[idx]).float()

        y = torch.from_numpy(np.array(self.labels[idx])).long()
        return (x, a, g, w), y


if __name__ == '__main__':
    path = "data/mit-bih/val/"
    ecg_dataset = Ecg_loader(path, "cwt")
    trainloader = data.DataLoader(ecg_dataset, batch_size=3, shuffle=True, num_workers=1)

    (x, a, g, w), y = next(iter(trainloader))
    print(x.shape)
    print(a.shape)
    print(g.shape)
    print(w.shape)
    print(y.shape)

