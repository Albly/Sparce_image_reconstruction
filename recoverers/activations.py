import torch

def threshold(v, lambd, type = 'soft'):
    '''
    Thresholding activation function. 
    if @type is 'soft' applies SoftThresholding function
    if @type is 'hard' applies HardThresholding fuction

    About functions
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.1124&rep=rep1&type=pdf
    '''

    zero = torch.tensor(0.0)            # zero scalar

    if type == 'soft':          
        tmp = torch.abs(v) - lambd
        return torch.sign(v)*torch.where(tmp<zero, zero, tmp)

    elif type == 'hard': 
        tmp = torch.abs(v) - lambd
        return torch.where(tmp<zero, zero, v)
    
    else:
        raise('No {} type for threshold function'.format(type))
        
    