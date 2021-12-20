import torch

# def threshold(v, lambd, type = 'soft'):
#     '''
#     Thresholding activation function. 
#     if @type is 'soft' applies SoftThresholding function
#     if @type is 'hard' applies HardThresholding fuction

#     About functions
#     https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.1124&rep=rep1&type=pdf
#     '''

#     zero = torch.tensor(0.0)            # zero scalar

#     if type == 'soft':          
#         tmp = torch.abs(v) - lambd
#         return torch.sign(v)*torch.where(tmp<zero, zero, tmp)

#     elif type == 'hard': 
#         tmp = torch.abs(v) - lambd
#         return torch.where(tmp<zero, zero, v)
    
#     else:
#         raise('No {} type for threshold function'.format(type))
        
    
def threshold(v, lambd, type = 'soft'):
    '''
    Thresholding activation function generalized to complex values. 
    if @type is 'soft' applies SoftThresholding function
    
    About functions
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.1124&rep=rep1&type=pdf
    '''

    if type != 'soft': 
        raise Exception('No {} type for threshold function'.format(type))

    zero = torch.zeros(v.shape)         # zero vector with same shape as input
    sgn = torch.sgn(v)                  # sgn of vector. Equivalent to sign if v - real

    return sgn*torch.maximum(torch.abs(v) - lambd , zero) 
