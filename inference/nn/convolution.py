import gc
import sys
import mmap
import numpy as np
import tenseal as ts
import multiprocessing as mp
import scipy.linalg as linalg
from memory_profiler import profile

# deprecated 請使用tenseal的新API
#@profile
def conv2d(tensor, conv_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        conv_parameters: return of "get_conv_para()"
    """
    kernels, bias, stride, padding = conv_parameters
            
    tensor_dim = tensor.shape
    kernel_dim = np.shape(kernels)[1:]
    
    tensor_c, tensor_h, tensor_w = tensor_dim[0], tensor_dim[1], tensor_dim[2]
    kernel_c, kernel_h, kernel_w = kernel_dim[0], kernel_dim[1], kernel_dim[2]
    
    if kernel_h<=padding[0]:
        print("padding[0] should be smaller than kernel height!")
        sys.exit()
        
    if kernel_w<=padding[1]:
        print("padding[1] should be smaller than kernel width!")
        sys.exit()
        
    if tensor_c != kernel_c:
        print("The kernel channel number does not match the tensor channel number!")
        sys.exit()
    
    context = tensor.context()
       
    result = []

    # loop over each kernel
    window = None
    for i, kernel in enumerate(kernels):
        kernel = np.array(kernel, dtype=np.float64)
        # loop over each convolution window
        for h in range(0-padding[0], tensor_h-kernel_h+1+padding[0], stride):
            for w in range(0-padding[1], tensor_w-kernel_w+1+padding[1], stride):
                #print(f"i={i}, h={h}, w={w}")
                h_start = -h if h<0 else 0
                w_start = -w if w<0 else 0
                h_end = tensor_h-h-kernel_h if h+kernel_h>tensor_h else kernel_h 
                w_end = tensor_w-w-kernel_w if w+kernel_w>tensor_w else kernel_w 

                kernel_ = kernel[:,h_start:h_end, w_start:w_end]
                kernel_ = ts.plain_tensor(kernel_)
                
                # flatten the kernel
                kernel_.reshape_([np.prod(kernel_.shape)])
                
                # sliding window
                window = tensor[0:tensor_c, max(h,0):min(h+kernel_h,tensor_h), max(w,0):min(w+kernel_w,tensor_w)]

                # flatten the window
                window.reshape_([np.prod(window.shape)])

                # do convolution
                window.dot_(kernel_)
                #window.mul_(kernel_).sum_(0).sum_(0).sum_(0)
                
                # collect the convolution result
                result.append(window.ciphertext()[0])
                
    context = window.context()                    
    del tensor

    output = ts.ckks_tensor( context=context, ciphertexts=result , shape=[len(result)])
    
    del result
    gc.collect()
    
    out_channel = len(kernels)
    out_height = int( (tensor_h + 2*padding[0] - kernel_h) / stride + 1 )
    out_width = int( (tensor_w + 2*padding[1] - kernel_w) / stride + 1 )
    
    if bias is not None:
        _bias = []
        for i in range(out_channel):
          _bias = _bias + [bias[i]] * (out_height*out_width)
        output.add_(_bias)

    return output.reshape_([out_channel, out_height, out_width])
    
#@profile
def conv2d_dw(tensor, conv_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        conv_parameters: return of "get_conv_dw_para()"
    """
    kernels, bias, stride, padding = conv_parameters
    
    tensor_dim = tensor.shape
    kernel_dim = np.shape(kernels)
    
    tensor_c, tensor_h, tensor_w = tensor_dim[0], tensor_dim[1], tensor_dim[2] 
    kernel_n, kernel_c, kernel_h, kernel_w = kernel_dim[0], kernel_dim[1], kernel_dim[2], kernel_dim[3]
    
    if kernel_h<=padding[0]:
        print("padding[0] should be smaller than kernel height!")
        sys.exit()
        
    if kernel_w<=padding[1]:
        print("padding[1] should be smaller than kernel width!")
        sys.exit()
    
    if kernel_c != 1:
        print("Every kernel channel number must equal to 1!")
        sys.exit()
        
    if tensor_c != kernel_n:
        print("The number of kernel does not match the tensor channel number!")
        sys.exit()
    
    context = tensor.context()
      
    result = []

    window = None
    # loop over each kernel
    for i, kernel in enumerate(kernels):
        kernel = np.array(kernel, dtype=np.float64)
        # loop over each convolution window
        for h in range(0-padding[0], tensor_h-kernel_h+1+padding[0], stride):
            for w in range(0-padding[1], tensor_w-kernel_w+1+padding[1], stride):
                #print(f"i={i}, h={h}, w={w}")
                h_start = -h if h<0 else 0
                w_start = -w if w<0 else 0
                h_end = tensor_h-h-kernel_h if h+kernel_h>tensor_h else kernel_h 
                w_end = tensor_w-w-kernel_w if w+kernel_w>tensor_w else kernel_w 

                kernel_ = kernel[:,h_start:h_end, w_start:w_end]
                kernel_ = ts.plain_tensor(kernel_)
                
                # flatten the kernel
                kernel_.reshape_([np.prod(kernel_.shape)])
                
                # sliding window
                window = tensor[i:i+1, max(h,0):min(h+kernel_h,tensor_h), max(w,0):min(w+kernel_w,tensor_w)]

                # flatten the window
                window.reshape_([np.prod(window.shape)])

                # do convolution
                window.dot_(kernel_)
                #window.mul_(kernel_).sum_(0).sum_(0).sum_(0)
                
                # collect the convolution result
                result.append(window.ciphertext()[0])

    context = window.context()       
    del tensor

    output = ts.ckks_tensor( context=context, ciphertexts=result , shape=[len(result)])
    
    del result
    gc.collect()
    
    out_channel = len(kernels)
    out_height = int( (tensor_h + 2*padding[0] - kernel_h) / stride + 1 )
    out_width = int( (tensor_w + 2*padding[1] - kernel_w) / stride + 1 )
    
    if bias is not None:
        _bias = []
        for i in range(out_channel):
          _bias = _bias + [bias[i]] * (out_height*out_width)
        output.add_( _bias )

    return output.reshape_( [out_channel, out_height, out_width] )
    

#@profile
def conv2d_as_matrix_mul(tensor, conv_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        conv_parameters: return of "get_conv_para()"
    """
    kernels, bias, stride, padding = conv_parameters
    
    b_K, c_K, h_K, w_K = np.shape(kernels)
    c_X, h_X, w_X = tensor.shape
    out_height = int((h_X+2*padding[0]-h_K)/stride+1)
    out_width = int((w_X+2*padding[1]-w_K)/stride+1)
    
    A, B = get_conv2d_matrix(tensor.shape, conv_parameters)
    
    tensor.reshape_([1, c_X*h_X*w_X])
    tensor.mm_(A.T)
    if B is not None:
        tensor.add_(B)
    
    return tensor.reshape_((b_K, out_height, out_width))


# refer to https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
# toeplitz_mult_ch
def get_conv2d_matrix(input_size, weights):
    """
    Args:
        input_size: (C_i, H_i, W_i)
    """
    kernels, bias, stride, pad = weights
    kernel_size = np.shape(kernels)
    k_b = kernel_size[0]
    k_c = kernel_size[1]
    k_h = kernel_size[2]
    k_w = kernel_size[3]
    i_c = input_size[0]
    i_h = input_size[1]
    i_w = input_size[2]
    o_h = int((i_h+2*pad[0]-k_h)/stride+1)
    o_w = int((i_w+2*pad[1]-k_w)/stride+1)
    
    output_size = (k_c, o_h, o_w)
    M = np.zeros((k_b, o_h*o_w, i_c, i_h*i_w))
    
    for i,ks in enumerate(kernels):
        for j,k in enumerate(ks):
            M_k = toeplitz_1_ch(k, (i_h, i_w), pad, stride)
            M[i, :, j, :] = M_k
            
    M.shape = (-1, i_c*i_h*i_w)
    
    N = None
    if bias is not None:
        N = []
        for i in range(k_b):
            N = N + [bias[i]] * (o_h*o_w)
        N = np.array(N, dtype=np.float64)
    
    return (M, N)
    

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
