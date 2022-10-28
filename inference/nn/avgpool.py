import gc
import sys
import math
import mmap
import numpy as np
import tenseal as ts
import multiprocessing as mp
from memory_profiler import profile    

#deprecated 請使用tenseal的API
#@profile
def avgpool2d(tensor, pool_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        pool_parameters: return of "get_pool_para()"
    """
    kernel_size, stride, padding, divisor = pool_parameters
    
    if kernel_size<=padding[0] or kernel_size<=padding[1]:
        print("Error: the padding size is larger than the kernel size!")
        sys.exit()

    tensor_dim = tensor.shape
    tensor_c, tensor_h, tensor_w = tensor_dim[0], tensor_dim[1], tensor_dim[2]
    
    context = tensor.context()
    
    result = []
    window = None
    for h in range(0-padding[0], tensor_h-kernel_size+1+padding[0], stride):
        for w in range(0-padding[1], tensor_w-kernel_size+1+padding[1], stride):
            # sliding window
            window = tensor[0:tensor_c, max(h,0):min(h+kernel_size,tensor_h), max(w,0):min(w+kernel_size,tensor_w)]
            
            # sum the window
            window.sum_(1).sum_(1)

            # collect the convolution result
            result.append(window.ciphertext())
    context = window.context()
    del tensor

    result = list(zip(*result))

    ciphertext = []
    for i in range(tensor_c):
        ciphertext.extend(result[i])
    
    del result
    gc.collect()
        
    out_height = int( (tensor_h + 2*padding[0] - kernel_size) / stride + 1 )
    out_width = int( (tensor_w + 2*padding[1] - kernel_size) / stride + 1 )
    new_shape = [tensor_c, out_height, out_width]

    if divisor==1:
        return ts.ckks_tensor( context=context, ciphertexts=ciphertext, shape=new_shape)
    else:
        return ts.ckks_tensor( context=context, ciphertexts=ciphertext, shape=new_shape).polyval([0,1/divisor])


#@profile
def avgpool2d_as_matrix_mul(tensor, pool_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        conv_parameters: return of "get_conv_para()"
    """
    kernel_size, stride, pad, divisor = pool_parameters
    
    h_K, w_K = kernel_size, kernel_size
    c_X, h_X, w_X = tensor.shape
    out_height = int((h_X+2*pad[0]-h_K)/stride+1)
    out_width = int((w_X+2*pad[1]-w_K)/stride+1)
    
    A, _ = get_avgpool2d_matrix_form(tensor.shape, pool_parameters)
    
    tensor.reshape_([1, c_X*h_X*w_X])
    tensor.mm_(A.T)

    return tensor.reshape_((c_X, out_height, out_width))


# refer to https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
# toeplitz_mult_ch
def get_avgpool2d_matrix(input_size, pool_parameters):
    """
    Args:
        kernel_size: (H_k, W_k)
        input_size : (n_in, H_i, W_i)
        pad        : (int, int)
        stride     : int
    """
    kernel_size, stride, pad, divisor = pool_parameters
    
    k_h, k_w = kernel_size, kernel_size
    i_c = input_size[0]
    i_h = input_size[1]
    i_w = input_size[2]
    o_h = int((i_h+2*pad[0]-k_h)/stride+1)
    o_w = int((i_w+2*pad[1]-k_w)/stride+1)
    
    kernels=[]
    for n in range(i_c):
        k = np.zeros((i_c,k_h,k_w), dtype=np.float64)
        k[n,:,:] = 1
        kernels.append(k.tolist())
    
    T = np.zeros((i_c, o_h*o_w, i_c, i_h*i_w), dtype=np.float64)
    
    for i,ks in enumerate(kernels):
        for j,k in enumerate(ks):
            T_k = toeplitz_1_ch(k, (i_h, i_w), pad, stride)
            T[i, :, j, :] = T_k
            
    T.shape = (-1, i_c*i_h*i_w)

    return (T*(1/divisor), None)
    

def multi_shift(first_row, shift_by, total_row):
    yield first_row
    while total_row > 1:
        first_row = np.append([0]*shift_by, first_row[:-shift_by])
        yield first_row
        total_row -= 1
        
        
# refer to https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
def toeplitz_1_ch(kernel, input_size, pad, stride):
    """
    Args:
        kernels   : shape=(H_k, W_k)
        input_size: (H_i, W_i)
        pad       : (int, int)
        stride    : int
    """
    # shapes
    k_h, k_w = np.shape(kernel)
    i_h, i_w = input_size
    o_h = int((i_h+2*pad[0]-k_h)/stride+1)
    o_w = int((i_w+2*pad[1]-k_w)/stride+1)

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz_shift = []
    for r in range(k_h):
        # len(first) == i_w + pad[1]
        first = [*kernel[r], *np.zeros(i_w-k_w+pad[1], dtype=np.float64)]
        # o_w = the number to shift right
        toeplitz_shift.append(np.stack(list(multi_shift(first, stride, o_w))))

    # delete front column
    if pad[1]:
        for r in range(k_h):
            n = pad[1]
            while n > 0:
                toeplitz_shift[r] = np.delete(toeplitz_shift[r],0,1)
                n = n-1
                
    #print(toeplitz_shift)
    
    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz_shift[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block), dtype=np.float64)

    for i, B in enumerate(toeplitz_shift):
        for j in range(o_h):
            if (i-pad[0])+j*stride >= 0 and (i-pad[0])+j*stride < w_blocks:
                W_conv[j, :, (i-pad[0])+j*stride, :] = B
            #print(W_conv.reshape((h_blocks*h_block, w_blocks*w_block)))
    
    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv