import sys
import torch
import numpy as np
import tenseal as ts
from memory_profiler import profile


def get_weight(layer_type, layer):
    if layer_type == "conv":
        out_channels, in_channels = layer.out_channels, layer.in_channels
        kernel_h, kernel_w = layer.kernel_size[0], layer.kernel_size[1]
        # must be 4 channel
        conv_weight = layer.weight.view( out_channels, in_channels, kernel_h, kernel_w ).tolist()
        conv_bias = None if layer.bias is None else layer.bias.tolist()
        conv_stride = layer.stride[0]
        conv_padding = layer.padding if type(layer.padding) == tuple else (layer.padding, layer.padding)
        return [conv_weight, conv_bias, conv_stride, conv_padding]
        
    elif layer_type == "bn" or layer_type == "bn1d" or layer_type == "bn2d":
        bn_gamma = layer.weight.tolist()
        bn_bias = layer.bias.tolist()
        bn_mean = layer.running_mean.tolist()
        bn_var = layer.running_var.tolist()
        bn_eps = layer.eps
        return [bn_gamma, bn_bias, bn_mean, bn_var, bn_eps]
    
    elif layer_type == "pool":
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding if type(layer.padding) == tuple else (layer.padding, layer.padding)
        divisor = layer.divisor_override 
        return [kernel_size, stride, padding, divisor]
    
    elif  layer_type == "linear":
        fc_weight = layer.weight.T.tolist()
        fc_bias = layer.bias.tolist()
        return [fc_weight, fc_bias]
        
    else:
        print(f"Error: no layer type {layer_type}.")
        sys.exit()

    
def merge(layer_list):
    A, B = layer_list[0]
    for M, N in layer_list[1:]:
        A = np.matmul(M, A)
        if B is None:
            B = N
        else:
            B = np.matmul(M, B)
            if N is not None:
                B = B + N
    return (A, B)


def matrix_to_filter(input_shape, matrix):
    """
    Args:
        input_shape    : (C, H, W)
        matrix         : shape = (H_m , C*H*W)
    """
    c_i, h_i, w_i = input_shape[0], input_shape[1], input_shape[2]
    h_m, w_m = matrix.shape[0], matrix.shape[1]
    
    channel_size = h_i*w_i
    if (c_i*channel_size) != w_m:
        print("Error: input_shape does not match the matrix shape!")
        sys.exit()
        
    kernel_list = []
    position_list = []
    for row in range(h_m):   
        m = matrix[row].reshape(input_shape)
        t, l = 0, 0
        d, r = h_i-1, w_i-1
        
        # find the first non-zero element index from the top
        while not np.any(m[:,t,:]):
            t = t + 1
        while not np.any(m[:,d,:]):
            d = d - 1
        while not np.any(m[:,:,l]):
            l = l + 1
        while not np.any(m[:,:,r]):
            r = r - 1
        
        if l > r or t > d:
            print("Error: empty kernel")
            sys.exit()
        
        kernel = m[:,t:d+1,l:r+1]
        
        found = False
        for idx, k in enumerate(kernel_list):
            if np.array_equal(kernel, k):
                found = True
                position_list.append((idx, t, l))
                break
                
        if not found:
            position_list.append((len(kernel_list), t, l))
            kernel_list.append(kernel)
            
    kernel_list = [ts.plain_tensor(k).data for k in kernel_list]      
    return position_list, kernel_list

 
# create TenSEAL context
#@profile
def get_context( level, poly_mod_degree=8192, bits_scale=25, integer=5 ):
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_mod_degree,
    coeff_mod_bit_sizes=[bits_scale+integer] + [bits_scale]*level + [bits_scale+integer])
    
    # set the scale
    context.global_scale = pow(2, bits_scale)
    
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context
