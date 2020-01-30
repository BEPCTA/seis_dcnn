#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a helper function DownsampleDataset(x,rate) 
which will apply downsampling to 3D tensor x and returns tuple of
downsampled tensor y and the original x. 
model = str --- saved pretrained model 
x = np.array[:,:,:] --- 3D array, the "input batch" 
rate = float --- parameter that specifies a type and degree of downsampling

"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import math, random
import glob
from scipy.interpolate import griddata

class DownsampleDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): batch of data
        rate: downsampling type
    """
    def __init__(self, xs, rate):
        super(DownsampleDataset, self).__init__()
        self.xs = xs
        self.rate = rate

    def __getitem__(self, index):
        batch_x = self.xs[index] #+ 10.
    
        if self.rate > 1 :
            mask = regular_mask(batch_x,self.rate)
        else:
            mask = random_mask(batch_x,self.rate)
           
        batch_y = mask.mul(batch_x)              
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)

def random_mask(data,rate):
  
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(),dtype=torch.float64) #tensor
    v = round(n*rate)
    columns_keep = random.sample(range(n),v)
    mask[:,:,columns_keep]=1  
    return mask

def regular_mask(data,a):

    """
    e.g: a = 5: 11110111101111
         a = 3: 10110110110 etc
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(),dtype=torch.float64)
    start = random.sample(range(a),1)[0]
    for i in range(n):
        mask[:,:,i] = (i + start)%a != 0   # to vary the pattern for the same a 
    return mask 


def gridInterpolation(d, m):
    x, y = np.indices(d.shape)
    interp = np.array(d)
    interp[m == 0] = griddata((x[m == 1], y[m == 1]), d[m == 1],       # points with values 
                              (x[m == 0], y[m == 0]), method='cubic')  # points to interpolate
    return interp

def mask(x, a):
    n = x.shape[1]
    m = np.zeros(x.shape)
    
    if a < 1:          # random stripes
        columns_keep = random.sample(range(n),round(a*n))
        m[:,columns_keep] = 1

    elif a > 1:   # regular stripes
        start = random.sample(range(a),1)[0]
        for i in range(n):
            m[:,i] = (i + start)%a != 0     # to vary the pattern for the same a 
    else:
        m[:,:] = 1
    return m
 
def masked(x, rate):
    mask = np.zeros(x.shape)
    if rate<1:          # reandom stripes
        TM = random.sample(range(x.shape[1]),round(rate*x.shape[1]))
        mask[:,TM] = 1

    elif rate>1:   # regular stripes
        for i in range(x.shape[1]):
            if (i+1)%rate==1:
                mask[:,i]=1
    else:
        return x
    mask = mask.astype(np.float64)
    return x*mask

def show3(x,y,x_):
    
    fig = plt.figure(figsize=(4, 13))
    plt.subplot(311)
    plt.imshow(x,vmin=-1, vmax=1, cmap="gray")
    plt.title('original')
    #plt.colorbar(shrink= 0.5)

    plt.subplot(312)   
    plt.imshow(y,vmin=-1, vmax=1, cmap="gray")
    plt.title('downsampled regularly')
    #plt.colorbar(shrink= 0.5)

    plt.subplot(313)   
    plt.imshow(x_,vmin=-1, vmax=1, cmap="gray")
    plt.title('downsampled randomly')
    #plt.colorbar(shrink= 0.5)
    plt.show()
 #   fig.savefig('report/report/fig2_h3.png', bbox_inches='tight')      

def read_mat(file):
    mat = loadmat(file)
    x0 = mat["img"][:, :150] 
    x1 = masked(x0, 2.0)
    x2 = masked(x0, 0.5)
    return x0, x1, x2

def read_matlab(mat_dir, size, n_data):
        
    file_list = glob.glob(mat_dir+'/*.mat')
    n = min(len(file_list), n_data)
    x = []
    for file in file_list[:n]:
        mat = loadmat(file)
        x.append(mat["img"][:size][:size])    
    return np.expand_dims(x, axis=3)

def rmse(y, x):
    return np.sqrt(((y - x) ** 2).mean())

