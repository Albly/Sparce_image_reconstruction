import torch
from torch import nn
import numpy as np
import recoverers.activations as act 


class Lista(torch.nn.Module):
    def __init__(self, A, layers, beta = 1.0,) -> None:
        super(Lista, self).__init__()
        INIT_LAMBDA = 0.001

        M,N = A.shape[0], A.shape[1]
        In = torch.eye(N)
        B = beta * torch.conj(A).T

        self.B = torch.nn.Parameter(B , requires_grad = True)
        self.S = torch.nn.Parameter(In-B@A, requires_grad=True)

        self.lambdas = torch.nn.Parameter(torch.ones(layers+1)*INIT_LAMBDA, requires_grad = True)      
        self.layers = layers
        
    def forward(self, y):
        By = self.B @ y 
        x_hat = act.threshold(By, lambd = self.lambdas[0], type= 'soft')

        for layer in range(self.layers):
            r = By + self.S @ x_hat
            x_hat = act.threshold(r, lambd=self.lambdas[layer+1])
        
        return x_hat
