# This file contains functions with metrics that are going to be used for quality estimation

import torch
from torch.nn.functional import mse_loss


def NMSE(x_real,x_hat):
    '''
    Normalized mean squared error in matrix form.
    '''

    return mse_loss(x_real, x_hat) / mse_loss(x_real, torch.zeros_like(x_real))


def PSNR(x_real,x_hat):
    '''
    Peak signal-to-noise ratio (PSNR)
    '''
    # get RMSE
<<<<<<< HEAD
    mse = torch.nn.MSELoss()
=======
    mse = mse_loss(x_real,x_hat)
>>>>>>> 1f734a55d0818cc8ef3d84bcd536eae4954d0682
    # maximum possible intensity for torch.uint8 dtype
    MAX_I = 255
    # calculate PSNR by definition
    PSNR = 10*torch.log10(MAX_I**2 / mse(x_real,x_hat))
    return PSNR