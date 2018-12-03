__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/03 19:33:25"

import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")
from models import *
from sys import exit
import argparse

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 1000
train_data = MNIST_Dataset(train_image)
train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)
test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)


vae = IWAE(50, 784)
vae.cuda()

optimizer = optim.Adam(vae.parameters())
num_epoches = 2
train_loss_epoch = []
for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        inputs = data.cuda()
        inputs = inputs.expand(5, batch_size, 784)
        
        optimizer.zero_grad()
        loss = vae.train_loss(inputs)
        loss.backward()
        optimizer.step()    
        print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}")
              .format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())


    # train_loss_epoch.append(np.mean(running_loss))

    # if (epoch + 1) % 1000 == 0:
    #     torch.save(vae.state_dict(),
    #                ("./output/model/{}_layers_{}_k_{}_epoch_{}.model")
    #                .format(args.model, args.num_stochastic_layers,
    #                        args.num_samples, epoch))
