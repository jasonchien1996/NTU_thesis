import gc
import sys
import math
import mmap
import ctypes
import numpy as np
import tenseal as ts
import multiprocessing as mp
from memory_profiler import profile

#@profile
def bn_polynomial_deg2(tensor, bn_parameters, coefficient):
    """
    Args:
        tensor       : shape=(n_in, H_i, W_i)
        bn_parameters: return of "get_bn_para()"
        coefficient  : [c, b, a] the coefficients of the polynomial of the form c + bx + ax^2
    """
    """
    bn: y = (x-mean)*weight/sqrt(var+eps) + bias = (x-E)*A+B = Mx + N 
    polynomial: y = c + b*x + a*x*x
    merge: y = a*N*N + b*N + c + M(2aN+b)*x + (a*M*M)*x*x = c0 + c1*x +c2*x*x 
    """
    weight, bias, mean, var, eps = bn_parameters
    c, b, a = coefficient[0], coefficient[1], coefficient[2]  
    tensor_c, tensor_h, tensor_w = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    
    if len(weight)!=tensor_c:
        print(f"Error: the tensor channel should be {len(mean)}!")
        sys.exit()
    
    channel_size = tensor_h * tensor_w
    
    c2 = []
    c1 = []
    c0 = []
    
    for i in range(tensor_c):
        M = weight[i]/(math.sqrt(var[i]+eps))
        N = bias[i] - mean[i]*M
        
        c2 = c2 + [a*M*M] * channel_size
        c1 = c1 + [M*(2*a*N+b)] * channel_size
        c0 = c0 + [a*N*N+b*N+c] * channel_size  

    c2 = np.reshape(c2, (tensor_c, tensor_h, tensor_w))
    c1 = np.reshape(c1, (tensor_c, tensor_h, tensor_w))
    c0 = np.reshape(c0, (tensor_c, tensor_h, tensor_w))

    tmp = tensor.copy()
    tensor.mul_(tensor).mul_(c2)
    tmp.mul_(c1).add_(c0)
    tensor.add_(tmp)
    
    return tensor