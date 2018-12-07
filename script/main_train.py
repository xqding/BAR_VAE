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
# sys.path.append("../model/")
# from iwae_models import *
from functions import *
from sys import exit
import argparse

## q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        logsigma = self.fc_logsigma(out)
        sigma = torch.exp(logsigma)
        return mu, sigma

## p(x|z)    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, output_dim),
                                       nn.Sigmoid())
    def forward(self, h):
        return self.transform(h)

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

## dimension of x, z, and hidden units
dim_z = 50
dim_x = 784
hidden_dim = 200

## each sample is associated with a set of particles z
## proprogated at different temperature
num_samples = train_image.shape[0]
num_z_p = 12
z_p_holder = np.random.randn(num_z_p, num_samples, dim_z)
z_p_holder = z_p_holder.astype(np.float32)

## encoder model q(z|x)
encoder = Encoder(dim_x, hidden_dim, dim_z)
optimizer_q = optim.Adam(encoder.parameters())

## decoder model p(x|z)
decoder = Decoder(dim_z, hidden_dim, dim_x)
optimizer_p = optim.Adam(decoder.parameters())

## inverse temperature
beta = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
beta = beta.reshape(-1,1)
beta = torch.ones(num_z_p, 1)

## energy and force on z under the generative model p(x,z) = p(z)*p(x|z)
def calc_energy_and_force(z, decoder, x, beta):
    for param in decoder.parameters():
        param.requires_grad = False
    x.requires_grad = False
    beta.requires_grad = False
    z.requires_grad = True
    
    prop = decoder(z)
    logP = torch.sum(-0.5*z**2, -1) + torch.sum(x*torch.log(prop) + (1-x)*torch.log(1-prop), -1)
    logP = beta * logP
    if z.grad is not None:
        z.grad.zero_()
    logP.backward(torch.ones(logP.shape))
    return -logP.data, z.grad.data

## Hamiltonian Monte Carlo
def HMC(current_q, epsilon, L, decoder, x, beta):
    q = torch.tensor(current_q.data, requires_grad = True)
    p = torch.randn(q.shape)
    current_p = torch.tensor(p.data, requires_grad = False)

    energy, force = calc_energy_and_force(q, decoder, x, beta)
    p = p + epsilon * force / 2

    for i in range(L):
        q.data = q.data + epsilon * p.data
        energy, force = calc_energy_and_force(q, decoder, x, beta)
        if i != L:
            p = p + epsilon * force
    p = p + epsilon * force / 2
    p = -p

    current_U, _ = calc_energy_and_force(current_q, decoder, x, beta)
    current_K = torch.sum(current_p**2, -1)/2

    proposed_U = energy
    proposed_K = torch.sum(p**2, -1)/2

    accept_ratio = torch.exp(current_U + current_K - proposed_U - proposed_K)
    flag = torch.rand(accept_ratio.shape) <= accept_ratio

    next_q = torch.tensor(current_q.data)
    next_q.data[flag] = q.data[flag]
    return next_q

## parameters for HMC
epsilon = 0.02
L = 50


mini_batch_size = 100
for epoch in range(10):
    ## shuffle index before processing each epoch
    random_idx = np.array(range(train_image.shape[0]))
    np.random.shuffle(random_idx)

    ## training
    for step in range(train_image.shape[0]//mini_batch_size):        
        ## select a minibatch of samples        
        idx = random_idx[step*mini_batch_size:(step+1)*mini_batch_size]

        x = np.random.binomial(1, train_image[idx, :]).astype('float32')
        x = torch.from_numpy(x)

        ####  training decoder p(x,z) ####
        
        ## samples from q(z|x)        
        num_z_q = 12
        mu, sigma = encoder(x.expand(mini_batch_size, dim_x))
        eps = sigma.new_empty(num_z_q, mini_batch_size, dim_z).normal_()
        z_q = mu + sigma * eps
        z_q = z_q.data

        ## run HMC to sample a new z_p from p(z|x)
        z_p = torch.tensor(z_p_holder[:, idx, :], requires_grad = True)
        z_p = HMC(z_p, epsilon, L, decoder, x, beta)
        z_p_holder[:, idx, :] = np.copy(z_p.numpy())

        ## put samples from q(z|x) and p(z|x) together
        z = torch.cat((z_q, z_p))

        ## calculated energies of each sample at states q(z|x) and p(x,z)
        energy_q = torch.sum(0.5*((z.data - mu.data)/sigma.data)**2 + 0.5*torch.log(sigma.data), -1)

        for param in decoder.parameters():
            param.requires_grad = True    
        prop = decoder(z) 
        energy_p = -(torch.sum(-0.5*z**2, -1) + torch.sum(x*torch.log(prop) + (1-x)*torch.log(1-prop), -1))

        ## concatnate energy_q and energy_p and use mbar to calculate the free energies
        energy = torch.stack((energy_q, energy_p), dim = 1)
        count = torch.tensor([[num_z_q],[num_z_p]], dtype = energy.dtype)
        count = count.expand(2,100)
        F, bias = calculate_free_energy_mbar(energy.data, count.data, verbose = True)

        ## calculate the ELBO for optimizing decoder
        biased_energy = energy.data + bias
        #biased_energy = biased_energy - torch.min(biased_energy.view(-1, mini_batch_size), 0)[0]
        log_weight = - energy_p.data - torch.log(torch.sum(torch.exp(-biased_energy), 1))
        log_weight_max = torch.max(log_weight, 0)[0]
        log_weight = log_weight - log_weight_max
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        loss_decoder = - torch.mean(torch.sum(weight * (-energy_p - torch.log(torch.sum(torch.exp(-biased_energy), 1))), 0))
        loss_decoder.backward()
        optimizer_p.step()
        loss_decoder_value = -torch.mean(torch.mean(torch.exp(log_weight), 0) + log_weight_max)

        #### training encoder ####
        ## calculate ELBO for optimizing encoder q(z|x)
        mu, sigma = encoder(x.expand(mini_batch_size, dim_x))
        #eps = sigma.new_empty(num_z_q, mini_batch_size, dim_z).normal_()
        z_q = mu + sigma * eps
        for param in decoder.parameters():
            param.requires_grad = False
        prop = decoder(z_q) 
        logP = torch.sum(-0.5*z_q**2, -1) + torch.sum(x*torch.log(prop) + (1-x)*torch.log(1-prop), -1)
        logQ = torch.sum(-0.5*eps**2 - torch.log(sigma), -1)
        log_weight = logP - logQ
        log_weight_max = torch.max(log_weight.data, 0)[0]
        weight = torch.exp(log_weight.data - log_weight_max)
        weight = weight / torch.sum(weight, 0)
        loss_encoder = -torch.mean(torch.sum(weight * log_weight, 0))
        loss_encoder.backward()
        optimizer_q.step()
        loss_encoder_value = -torch.mean(torch.mean(torch.exp(log_weight.data - log_weight_max), 0) + log_weight_max)

        ## log information
        print("Epoch: {:>3d}, Step: {:>5d}, loss_decoder: {:>10.3f}, loss_encoder: {:>10.3f}".format(epoch, step, loss_decoder_value, loss_encoder_value))
