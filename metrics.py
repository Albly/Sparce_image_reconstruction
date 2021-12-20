# This file contains functions with metrics that are going to be used for quality estimation

import torch
from torch.nn.functional import mse_loss
import numpy as np
import cv2

def MSE(x_real, x_hat):
    '''Mean squared error generalized for complex values'''

    assert x_real.shape == x_hat.shape, 'Sizes of both values must be the same'
    
    if torch.is_complex(x_real):
        mse = torch.sum(torch.abs(x_real-x_hat)**2)/torch.numel(x_real)

    else:
        mse = mse_loss(x_real, x_hat)

    return mse


def NMSE(x_real,x_hat):
    '''
    Normalized mean squared error in matrix form.
    '''
    nmse = MSE(x_real, x_hat) / MSE(x_real, torch.zeros(x_real.shape))
    return nmse.item()


def PSNR(x_real,x_hat):
    '''
    Peak signal-to-noise ratio (PSNR)
    '''
    # get RMSE
    mse = mse_loss(x_real,x_hat)

    # maximum possible intensity for torch.uint8 dtype
    MAX_I = 255
    # calculate PSNR by definition
    PSNR = 10*torch.log10(MAX_I**2 / mse)
    return PSNR.item()




def SSIM(img1, img2):
    '''
    Structural similarity index measure (SSIM) 
    '''
    # USEFUL LINKS
    # https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    # https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
    # https://www.tensorflow.org/api_docs/python/tf/image/ssim?hl=ru
    # L -dynamic pyxel range 
    
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)
    
    L,k1,k2 = 255,0.01,0.03
    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    
    # create kernel
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    # calculate mean values
    muX = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    muY = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    muX_sq,muY_sq = muX**2,muY**2 
    muXY = muX * muY
    
    # calculate deviations
    sigmaX_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - muX_sq
    sigmaY_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - muY_sq
    sigmaXY = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - muXY
    
    # calculate SSIM over xy plane
    ssim_map = ((2 * muXY + C1) * (2 * sigmaXY + C2)) / ((muX_sq + muY_sq + C1) * (sigmaX_sq + sigmaY_sq + C2))
    
    # average results
    ssim = ssim_map.mean()
    return ssim