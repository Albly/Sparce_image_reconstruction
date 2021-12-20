import numpy as np
import pywt
import matplotlib.pyplot as plt
from itertools import accumulate 

def wlet_forward(image,wlet,levl):
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

def wlet_inverse(s_w,cMat_shapes,wlet,levl):
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