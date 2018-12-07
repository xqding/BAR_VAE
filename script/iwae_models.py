__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/03 19:14:35"

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')

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

    
class IWAE(nn.Module):
    def __init__(self, dim_z, dim_image_vars):
        super(IWAE, self).__init__()
        self.dim_z = dim_z
        self.dim_image_vars = dim_image_vars

        ## encoder
        self.encoder_z = Encoder(dim_image_vars, 200, dim_z)
        
        ## decoder
        self.decoder_x = Decoder(dim_z, 200, dim_image_vars)
        
    def encoder(self, x):
        mu_z, sigma_z = self.encoder_z(x)
        eps = Variable(sigma_z.data.new(sigma_z.size()).normal_())
        z = mu_z + sigma_z * eps                
        return z, mu_z, sigma_z, eps
    
    def decoder(self, z):
        p = self.decoder_x(z)
        return p
    
    def forward(self, x):
        z, mu_z, sigma_z, eps = self.encoder(x)
        p = self.decoder(z)
        return (z, mu_z, sigma_z, eps), (p)

    def train_loss(self, inputs):
        z, mu_z, sigma_z, eps = self.encoder(inputs)
        log_QzGx = torch.sum(-0.5*(eps)**2 - torch.log(sigma_z), -1)
        
        p = self.decoder(z)
        log_Pz = torch.sum(-0.5*z**2, -1)
        log_PxGz = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)

        log_weight = log_Pz + log_PxGz - log_QzGx
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = Variable(weight.data, requires_grad = False)
        loss = -torch.mean(torch.sum(weight * (log_Pz + log_PxGz - log_QzGx), 0))
        return loss

    def test_loss(self, inputs):
        z, mu_z, sigma_z, eps = self.encoder(inputs)
        log_QzGx = torch.sum(-0.5*(eps)**2 - torch.log(sigma_z), -1)
        
        p = self.decoder(z)
        log_Pz = torch.sum(-0.5*z**2, -1)
        log_PxGz = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)

        log_weight = log_Pz + log_PxGz - log_QzGx
        weight = torch.exp(log_weight)
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))        
        return loss
