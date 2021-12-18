# This file contains functions with metrics that are going to be used for quality estimation

import torch

# def MSE(x_real,x_hat):
#     '''
#     Mean squared error metric in matrix form
#     Calculates MSE via Frobenius norm.  
#     '''

#     #input shapes
#     x_real_shape = x_real.shape
#     x_hat_shape = x_hat.shape

#     # input shapes must be equal
#     assert x_real_shape == x_hat_shape, 'Inputs should be the same size. got {0} and {1}'.format(x_real_shape, x_hat_shape)

#     return torch.sum(torch.abs(x_real-x_hat)**2)/torch.numel(x_real)



def NMSE(x_real,x_hat):
    '''
    Normalized mean squared error in matrix form.
    '''
    mse = torch.nn.MSELoss()
    return mse(x_real, x_hat) / mse(x_real, torch.zeros_like(x_real))



def PSNR(x_real,x_hat):
    '''
    Peak signal-to-noise ratio (PSNR)
    '''
    # get RMSE
    mse = torch.nn.MSELoss()
    # maximum possible intensity for torch.uint8 dtype
    MAX_I = 255
    # calculate PSNR by definition
    PSNR = 10*torch.log10(MAX_I**2 / mse(x_real,x_hat))
    return PSNR