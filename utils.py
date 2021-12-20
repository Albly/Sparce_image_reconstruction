import scipy
from scipy import linalg
import torch
import math
import numpy as np

import pywt
from itertools import accumulate 

def toDb(value):
    '''Transforms ratio to Decibels'''
    return 10*torch.log10(value)


def fromDb(value):
    '''Transforms Decibels to ratio'''
    return 10**(value/10.0)


def get_noise(signal, SNR_dB):
    '''
    Generates Additive white Gaussian noise (AWGN) with the same shape
    as shape of input @signal. Power of noise is calculsted by given
    @SNR_db signal to noise ratio [dB].  
    '''
    
    P_signal = torch.mean(torch.abs(signal)**2)         # calculate power of the signal [W]
    P_noise = P_signal*10**(-0.1*SNR_dB)                # calculate power of noise [W]
    D_noise = torch.sqrt(P_noise)                       # Deviation of AWGN
    noise = D_noise*torch.randn_like(signal)            # AWGN with zero mean and @D_noise std

    return noise


# MEASUREMENT MATRIX GENERATORS
#=================================================================================================

def get_partial_FFT(M,N):
    '''
    Retuns MxN partial normalized DFT matrix with
    Restricted Isometry Property (RIP)
    '''
    assert M < N , 'M should be less than N'
    F = torch.tensor(scipy.linalg.dft(N)/math.sqrt(N))  # DFT matrix
    idxs = torch.randperm(N)[:M]                        # randomly choose M rows from N-size DFT matrix
    return F[idxs,:]


def get_Gaussian_matrix(M,N):
    '''
    Retuns MxN partial normalized random matrix with
    Gaussian distributed values and
    Restricted Isometry Property (RIP)
    '''
    assert M < N , 'M should be less than N'
    
    A = np.random.normal(size=(M,N), scale = 1.0/np.sqrt(M)).astype(np.float32) # random matrix 
    A = A/np.linalg.norm(A,2)   # normalize to max singular value
    return A

#=================================================================================================


def wlet_forward(image,wlet='bior1.3',levl=1):
    '''
    image - image to be transformed
    wlet - wawelet type
    levl - level of wawelet transform
    '''
    # get i-th level coefficients
    coeffs = pywt.wavedec2(image, wavelet=wlet, level=levl)
    cMat_list = [coeffs[0]]
    for c in coeffs[1:]:
        cMat_list = cMat_list + list(c)
    cMat_shapes = list(map(np.shape,cMat_list))
    vect = lambda array: np.array(array).reshape(-1)
    cVec_list = [vect(cMat_list[j]) for j in range(3*levl+1)]
    s_w = np.concatenate(cVec_list)
    return s_w,cMat_shapes

def wlet_inverse(s_w,cMat_shapes,wlet='bior1.3',levl=1):
    '''
    s_w - wawelet vector to be transformed
    cMat_shapes of arrays of coefficients
    wlet - wawelet type
    levl - level of wawelet transform
    '''
    cVec_shapes = list(map(np.prod,cMat_shapes))     
    split_indices = list(accumulate(cVec_shapes))
    cVec_list = np.split(s_w,split_indices)
    cVec_list = [cVec_list[j] for j in range(3*levl+1)]
    coeffs=[ np.reshape(cVec_list[0],cMat_shapes[0]) ]
    for j in range(levl):
        triple = cVec_list[3*j+1:3*(j+1)+1]
        triple = [np.reshape( triple[i], cMat_shapes[1 +3*j +i] ) for i in range(3) ]
        coeffs = coeffs + [tuple(triple)]
    # get image from coefficients
    image_rec = pywt.waverec2(coeffs, wavelet=wlet)
    return image_rec