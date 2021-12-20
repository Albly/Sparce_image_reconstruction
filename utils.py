import scipy
from scipy import linalg
import torch
import math
import numpy as np

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

