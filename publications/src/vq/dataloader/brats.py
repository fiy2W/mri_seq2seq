import os
import numpy as np
import SimpleITK as sitk
import random

import torch
from torch.utils.data import Dataset


class Dataset_brats(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.root = args['data']['image']

        self.file_X = []
        self.indicator = []

        with open(args['data'][mode], 'r') as f:
            strs = f.readlines()
            for line in strs:
                name = line.split(',')[0]
                indicator = bin(np.int32(line.split(',')[1]))[2:].zfill(4)
                self.file_X.append(os.path.join(self.root, name))
                if self.mode=='train':
                    num_i = np.int32(line.split(',')[1])
                    if num_i%3==0:
                        self.indicator.append('1100')
                    elif num_i%3==1:
                        self.indicator.append('0110')
                    elif num_i%3==2:
                        self.indicator.append('0011')
                else:
                    self.indicator.append(indicator)
    
    def __len__(self):
        return len(self.file_X)
    
    def norm(self, arr):
        """ norm (0, 99%*2) to (-1, 1)
        arr: [s,d,w,h]
        """
        amax = np.percentile(arr, 99) * 2
        arr = np.clip(arr, 0, amax) / amax * 2 - 1
        return arr
    
    def preprocess(self, x, k=[0,0,0], axis=[0,1,2]):
        nd = 128
        d, w, h = x.shape
        rd = (d-nd)//2 if d>nd else 0
        x = x[rd:rd+nd]

        if self.mode=='train':
            x = np.transpose(x, axes=axis)
            if k[2]==1:
                x = x[:, :, ::-1]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[0]==1:
                x = x[::-1, :, :]
            x = x.copy()

        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0) # [1,1,d,w,h]
        return x
    
    def __getitem__(self, index):
        x_idx = self.file_X[index]
        i_idx = self.indicator[index]

        imgs_t1 = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_t1.nii.gz'.format(os.path.basename(x_idx))))))
        imgs_t1ce = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_t1ce.nii.gz'.format(os.path.basename(x_idx))))))
        imgs_t2 = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_t2.nii.gz'.format(os.path.basename(x_idx))))))
        imgs_flair = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_flair.nii.gz'.format(os.path.basename(x_idx))))))
        segs = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_seg.nii.gz'.format(os.path.basename(x_idx)))))

        k = [0, 0, random.randint(0,1)]
        axis = [0,1,2]
        imgs_t1 = self.preprocess(imgs_t1, k, axis)
        imgs_t1ce = self.preprocess(imgs_t1ce, k, axis)
        imgs_t2 = self.preprocess(imgs_t2, k, axis)
        imgs_flair = self.preprocess(imgs_flair, k, axis)
        segs = self.preprocess(segs, k, axis)

        ind = []
        for i in range(4):
            if i_idx[i] == '1':
                ind.append(i)
        
        return {
            't1': torch.from_numpy(imgs_t1),
            't1ce': torch.from_numpy(imgs_t1ce),
            't2': torch.from_numpy(imgs_t2),
            'flair': torch.from_numpy(imgs_flair),
            'segs': torch.from_numpy(np.array(segs, np.uint8)),
            'flag': ind,
            'path': [x_idx],
        }