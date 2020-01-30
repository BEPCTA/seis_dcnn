#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a helper function predict(x,model) 
which will apply DnCNN model for prediction on x and returns the result.
model = str --- saved pretrained model 
x = np.array[:,:] --- 2D array, the "input" 
"""

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch

class DnCNN(nn.Module):
    def __init__(self, n_layers = 17, n_channels=64, image_channels = 1,
                 use_bnorm = True, kernel_size = 3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, 
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('weights initialized')

def predict(x, model = None):
    if model == None: 
        return x
    cpu_model = torch.load(model, map_location=lambda storage, loc: storage)
    cpu_model.eval() 
    y = cpu_model(torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]))
    y = y.view(x.shape[0], x.shape[1]) 
    y = y.cpu()
    y = y.detach().numpy().astype(np.float32)
    return y

if __name__ == '__main__':
#    from scipy.io import loadmat
#    from read_matlab import masked
#    from read_matlab import show3
#    matfile= "data/validation/saltdome_0167.mat"
#    trained_model = "saved_models/model_040.pth"
##    mat = loadmat(matfile)
##    x0 = mat["img"][:, :] 
##    x1 = masked(x0, 2.0)
##    x2 = masked(x0, 0.5)
##    show3(x0, x1, x2)
#    show3(x0, predict(x1, trained_model),  predict(x2, trained_model))
#    
     print("Hello, reviewer !")