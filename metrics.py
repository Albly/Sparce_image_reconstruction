# This file contains functions with metrics that are going to be used for quality estimation

import torch

def MSE(x_real,x_hat):
    x_real_shape = x_real.shape
    x_hat_shape = x_hat.shape

    assert x_real_shape == x_hat_shape, 'Inputs should be the same size. got {0} and {1}'.format(x_real_shape, x_hat_shape)

    if len(x_real_shape) == 1:
        x_real = x_real.reshape(1,-1)
        x_hat = x_hat.reshape(1,-1)
    
    return torch.norm(x_hat-x_hat, 'fro')**2 / torch.numel(x_real)



def NMSE(x_real,x_hat):

    return MSE(x_real, x_hat) / MSE(x_real, torch.zeros_like(x_real))

