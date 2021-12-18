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
    '''

    v = y - A @ x_hat
    r = x_hat + A.T @ v
    x_hat = act.threshold(r, lambd, 'soft')
    return x_hat


def ista_denoise(y,A,x, lambd, iterations):
    
    x_hat = torch.zeros(A.shape[1], 1)
    metric = []

    for i in range(iterations):
        x_hat = ista_iteration(y,A, x_hat ,lambd)
        metric.append(NMSE(x,x_hat))

    return x_hat, torch.tensor(metric)

#=========================================================================

def fista_iteration(y, A, x_hat, x_hat_prev, lambd, t):
    v = y - A@x_hat
    r_add = (t-2)/(t+1) * (x_hat - x_hat_prev)
    r = x_hat + A.T @ v + r_add
    x_hat_prev = x_hat
    x_hat = act.threshold(r, lambd, 'soft')

    return x_hat, x_hat_prev

def fista_denoise(y,A,x, lambd, iterations):
    x_hat = torch.zeros(A.shape[1], 1)
    x_hat_prev = torch.zeros(A.shape[1], 1)
    metric = []

    for t in range(1, iterations):
        x_hat, x_hat_prev = fista_iteration(y,A, x_hat, x_hat_prev, lambd, t)  
        metric.append(NMSE(x,x_hat))
    return x_hat, torch.tensor(metric)
    
#=========================================================================

def amp_iteration(y,A, x_hat, alpha, v_prev):
    
    b = 1/A.shape[0] * len(x_hat[x_hat!=0])

    v = y - A @x_hat + b*v_prev
    lambd = alpha/math.sqrt(A.shape[0]) * torch.norm(v,2)
    r = x_hat + A.T @ v
    x_hat = act.threshold(r,lambd, 'soft')

    v_prev = v 

    return x_hat, v_prev

def amp_denoise(y,A, x , alpha, iterations):
    x_hat = torch.zeros(A.shape[1],1)
    v_prev = torch.zeros(A.shape[0],1)
    metric = []

    for _ in range(iterations):
        x_hat, v_prev = amp_iteration(y,A, x_hat,alpha, v_prev )
        metric.append(NMSE(x,x_hat))

    return x_hat, torch.tensor(metric)
