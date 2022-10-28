import sys
import math
import numpy as np
import tenseal as ts
from memory_profiler import profile


#@profile
def batchnorm2d(tensor, bn_parameters):
    """
    Args:
        tensor       : shape=(n_in, H_i, W_i)
        bn_parameters: return of "get_bn_para()"
    """
    weight, bias, mean, var, eps = bn_parameters
    bn_c = len(weight)
    tensor_c, tensor_h, tensor_w = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    
    if bn_c != tensor_c:
        print("The tensor channel does not match the batchnorm channel!")
        sys.exit()
    
    channel_size = tensor_h*tensor_w
    
    _weight = []
    _bias = []
    
    for i in range(bn_c):
        W = weight[i]/(math.sqrt(var[i]+eps))
        _weight = _weight + [W]*channel_size
        _bias = _bias + [bias[i]-mean[i]*W]*channel_size

    _weight = np.reshape(_weight, (tensor_c, tensor_h, tensor_w))
    _bias = np.reshape(_bias, (tensor_c, tensor_h, tensor_w))
    tensor.mul_(_weight).add_(_bias)
    return tensor


#@profile    
def get_bn2d_matrix(input_size, weights):
    """
    Args:
        input_size   : shape=(n_in, H_i, W_i)
        weights: return of "get_bn_para()"
    """
    weight, bias, mean, var, eps = weights
    bn_c = len(weight)
    i_c, i_h, i_w = input_size[0], input_size[1], input_size[2]
    
    if bn_c != i_c:
        print("The tensor channel does not match the batchnorm channel!")
        sys.exit()
    
    vector_len = i_c*i_h*i_w
    A = np.zeros((vector_len,vector_len), dtype=np.float64)
    B = np.zeros(vector_len, dtype=np.float64)
    for i in range(i_c):
        index = np.arange(i*i_h*i_w, (i+1)*i_h*i_w)
        W = weight[i]/(math.sqrt(var[i]+eps))
        A[index,index] = W
        B[index] = bias[i]-W*mean[i]
    return (A, B)
    
#@profile    
def get_bn1d_matrix(input_size, bn_parameters):
    """
    Args:
        input_size   : int
        bn_parameters: return of "get_bn_para()"
    """
    weight, bias, mean, var, eps = bn_parameters
    
    if input_size != len(weight):
        print("Error: input_size does not match the parameters!")
        
    A = np.zeros((input_size,input_size), dtype=np.float64)
    B = np.zeros(input_size, dtype=np.float64)
    for i in range(input_size):
        W = weight[i]/(math.sqrt(var[i]+eps))
        A[i,i] = W
        B[i] = bias[i]-W*mean[i]
    return (A, B)