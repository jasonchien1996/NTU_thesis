import time
import numpy as np
import tenseal as ts
from memory_profiler import profile


#@profile
def linear(tensor, linear_parameters):
    """
    Args:
        tensor         : shape=(n_in, H_i, W_i)
        linear_parameters: return of "get_linear_para()"
    """
    weight, bias = linear_parameters
    tensor_size = np.prod(tensor.shape)
    
    # to use mm_() tensor must be 2d
    #t1 = time.time()
    tensor.reshape_([1, tensor_size])
    #t2 = time.time()
    #print(f"==> reshape {t2-t1} seconds")
    
    #t3 = time.time()
    tensor.mm_(weight)
    #t4 = time.time()
    #print(f"==> mm {t4-t3} seconds")
    
    #t5 = time.time()
    tensor.add_(bias)
    #t6 = time.time()
    #print(f"==> add {t6-t5} seconds")
    
    #print(f"==> total {t6-t1} seconds")
    
    return tensor    
    
def get_linear_matrix(linear_parameters):
    weight, bias = linear_parameters
    if bias is not None:
        return (np.array(weight, dtype=np.float64).T, np.array(bias, dtype=np.float64))
    return (np.array(weight, dtype=np.float64).T, None)