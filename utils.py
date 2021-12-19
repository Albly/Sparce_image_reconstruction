import torch

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
    
    P_signal = torch.mean(torch.abs(signal)**2) # calculate power of the signal [W]
    P_noise = P_signal*10**(-0.1*SNR_dB) # calculate power of noise [W]
    D_noise = torch.sqrt(P_noise) # Deviation of AWGN
    noise = D_noise*torch.randn_like(signal) # noise

#     P_signal = torch.mean(torch.abs(signal)**2) # calculate power of the signal [W]
#     D_noise = P_signal/fromDb(SNR_dB)           # calculate power of noise [W]
    
#     mu = torch.zeros(1)                         # Mean of AWGN 
#     sigma = torch.sqrt(D_noise)               # Deviation of AWGN

#     shape = signal.shape                        # get shape of image
#     noise = torch.normal(mean = mu.repeat(shape), std = sigma.repeat(shape)) # Noise vector
    return noise

