from typing import Match
import torch
from recoverers import activations as act
from tqdm.notebook import tqdm
from metrics import NMSE
import math


#=========================================================================
def ista_iteration(y, A, x_hat, lambd):
    '''
    iteration of Iterative Soft-Thresholding algorithm (ISTA)
    @y - noisy vector 
    @A - measurements matrix
    @x_hat - current extimation of original vector x
    @lamd - threshold for activation

    Baseline: https://arxiv.org/abs/1607.05966
    '''

    v = y - A @ x_hat                       # residual measurement error                        
    r = x_hat + torch.conj(A).T @ v         # input of thresholding function
    x_hat = act.threshold(r, lambd, 'soft') # apply thresholding

    return x_hat                    


def ista_denoise(y,A,x, lambd, iterations):
    '''
    Iterative Soft-Thresholding algorithm (ISTA)
    @y - noisy vector 
    @A - measurements matrix 
    @x - real vector x (for comparing)
    @lambd - threshold for activation function
    @iterations - number of ISTA iterations

    Baseline: https://arxiv.org/abs/1607.05966
    '''

    x_hat = torch.zeros(A.shape[1], 1, dtype = x.dtype) # init estimated x as zero vec
    metric = []                                         # NMSE list

    for i in range(iterations):                         
        x_hat = ista_iteration(y,A, x_hat ,lambd)       # do ista iterations
        metric.append(NMSE(x, x_hat))                   # calculate nmse

    return x_hat, torch.tensor(metric)


#=========================================================================

def fista_iteration(y, A, x_hat, x_hat_prev, lambd, t):
    '''
    Iteration of Fast Iterative Soft-Thresholding algorithm (FISTA).
    @y -           noisy vector 
    @A -           measurements matrix
    @x_hat -       current estimation of original vector x
    @x_hat_prev -  x_hat estimation from previous iteration
    @lambd -       threshold for activation
    @t -           number of current iteration

    Baseline: https://arxiv.org/abs/1607.05966
    '''

    v = y - A @ x_hat                               # residual error
    r_add = (t-2)/(t+1) * (x_hat - x_hat_prev)      # additional term, that makes convergence faster
    r = x_hat + torch.conj(A).T @ v + r_add         # input of activation
    x_hat_prev = x_hat                              
    x_hat = act.threshold(r, lambd, 'soft')         # activation     

    return x_hat, x_hat_prev

def fista_denoise(y,A,x, lambd, iterations):
    '''
    Fast Iterative Soft-Thresholding algorithm (FISTA).
    @y -      noisy vector 
    @A -      measurements matrix
    @x -      original vector x. (For metric calculating)
    @lambd -  threshold for activation
    @iterations - number of fista iterations

    Baseline: https://arxiv.org/abs/1607.05966
    '''

    x_hat = torch.zeros(A.shape[1], 1, dtype = x.dtype)         # init current estimation of x_hat as zeros 
    x_hat_prev = torch.zeros(A.shape[1], 1, dtype = x.dtype)    # the same with previous estimation
    metric = []

    for t in range(1, iterations):              
        x_hat, x_hat_prev = fista_iteration(y,A, x_hat, x_hat_prev, lambd, t) # do iterations 
        metric.append(NMSE(x,x_hat))        # calculate NMSE

    return x_hat, torch.tensor(metric)

#=========================================================================

def amp_iteration(y,A, x_hat, alpha, v_prev):
    '''
    Iteration of Approimate message passing algorithm. (AMP)
    @y -      noisy vector 
    @A -      measurements matrix
    @x_hat -  current estimation of original vector x
    @alpha -  threshold for activation
    @v_prev - residual error from previous iteration
    '''
    
    b = 1/A.shape[0] * len(x_hat[x_hat!=0])                 # Onsager correction for soft-shrinkage func

    v = y - A @x_hat + b*v_prev                             # residual error
    lambd = alpha/math.sqrt(A.shape[0]) * torch.norm(v,2)   # calculate threshold
    r = x_hat + torch.conj(A).T @ v                         # input for activation      
    x_hat = act.threshold(r,lambd)                          # activation               
    v_prev = v                      

    return x_hat, v_prev


def amp_denoise(y,A, x , alpha, iterations):
    '''
    Approimate message passing algorithm.(AMP)
    @y -      noisy vector 
    @A -      measurements matrix
    @x -      original vector x. (For metric calculating)
    @alpha -  coeff for activation threshold  calculation
    @iterations - number of fista iterations

    Baseline: https://arxiv.org/abs/1607.05966
    '''

    x_hat = torch.zeros(A.shape[1],1, dtype= x.dtype)   #  initial x_hat - zero vec
    v_prev = torch.zeros(A.shape[0],1, dtype= x.dtype)  # prevous x_hat - zero vec
    metric = []

    for _ in range(iterations):
        x_hat, v_prev = amp_iteration(y,A, x_hat, alpha, v_prev ) # do amp iterations
        metric.append(NMSE(x, x_hat))                               # calcualte metric

    return x_hat, torch.tensor(metric)
