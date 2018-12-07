__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/06 03:16:04"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class mbar_loss(nn.Module):
    def __init__(self, energy, count, bias = None):
        """ Initializer for the class mbar_loss.

        Args:
            energy (Tensor): unitless energy values. It should have a size of
                             (num_configurations, num_states, batch_size)
            count (Tensor): num of configurations from each state. Size of it 
                            should be (num_states, batch_size)
            bias (None or Tensor): biasing energy to solve. It has a size of 
                                   (num_states, batch_size)

        Returns:
            loss (Tensor): the loss of mbar objective function
        
        """
        super(mbar_loss, self).__init__()
        self.energy = energy
        self.num_configurations = energy.shape[0]
        self.count = count

        if bias is None:
            bias = torch.tensor(-torch.min(self.energy, 0)[0],
                                device = self.energy.device)
            self.bias = nn.Parameter(bias)
        else:
            bias = torch.tensor(bias, device = self.energy.device,
                                requires_grad = True)
            self.bias = nn.Parameter(bias)
    
    def forward(self):
        """ Calculate the loss of mbar objective function

            The objection function is a convex function of the variable bias.
            The formula of the objective function can be found in the FastMBAR
            paper.
           
        Returns:
            loss (Tensor): the value of mbar loss function

        """
        loss = torch.exp(-(self.energy + self.bias))
        loss = torch.sum(torch.log(torch.sum(loss, 1)), 0)
        loss = loss + torch.sum(self.count*self.bias, 0)
        loss = 1.0/(self.num_configurations)*loss
        loss = torch.sum(loss)
        return loss

def calculate_free_energy_mbar(energy, count, bias = None, verbose = False):
    """ Function to calculate the free energies of states 

    This function calcualtes the free energies of states by minimizing the mbar
    loss function with the optimizer torch.optim.lbfgs

    Args:
        energy (Tensor): unitless energy values. It should have a size of
                         (num_configurations, num_states, batch_size)
        count (Tensor): num of configurations from each state. Size of it 
                        should be (num_states, batch_size)
        bias (None or Tensor): biasing energy to solve. It has a size of 
                               (num_states, batch_size)
        
    Returns:
        F (Tensor): unitless free energies. (num_states, batch_size)
        bias (Tensor): corresponding bias energyies. (num_states, batch_size)

    """    
    num_configurations = energy.shape[0]
    
    ## mbar_loss object to calcualte loss and grad
    loss_model = mbar_loss(energy, count, bias)
    optimizer = optim.LBFGS(loss_model.parameters(), max_iter = 10, tolerance_change=1e-5)

    ## calcualte loss and grad
    previous_loss = loss_model()
    previous_loss.backward()
    previous_loss = previous_loss.item()
    grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()

    ## minimize loss using L-BFGS-B
    if verbose:
        print("start loss: {:>7.5f}, start grad: {:>7.5f}".format(previous_loss, grad_max)) 
    for i in range(30):
        def closure():
            optimizer.zero_grad()
            loss = loss_model()
            loss.backward()    
            return loss
        optimizer.step(closure)
        loss = loss_model().item()
        grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()
        
        if verbose:
            print("step: {:>4d}, loss:{:>7.5f}, grad: {:>7.5f}".format(i, loss, grad_max))
            
        ## stop criterion for L-BFGS-B
        ## this is added because the optim.LBFGS often returns nan values
        ## when it runs too many iterations.
        if np.abs(loss-previous_loss) <= 1e-4 or grad_max <= 1e-4:
            break
        previous_loss = loss

    ## using the bias energies to calculate free energies for states with
    ## nonzero samples
    bias = loss_model.bias.data
    tmp = -torch.log(count/num_configurations)
    F = tmp - bias

    # # ## normalize free energyies of states with nonzero samples
    # F = F - torch.min(F,0)[0]
    # prob = torch.exp(-F)
    # prob = prob / prob.sum(-1, keepdim = True)
    # F_nz = - torch.log(prob)

    # ## update bias energy based on normalized F for states with
    # ## nonzero samples
    # bias = -torch.log(count/num_configurations) - F_nz

    return F, bias
