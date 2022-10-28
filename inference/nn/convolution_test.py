import time
import torch
import numpy as np
import tenseal as ts
from test_util import *
from convolution import *

np.set_printoptions(6, suppress=True, linewidth=120)


print("==> prepare context...")
context = get_context(1,8192,40)

# generate data
print("==> prepare data...")
channel = 1
height = 5
width = 5
# plain data
x = torch.from_numpy(np.random.random((1,channel,height,width))).float()
# encrypted data
enc_x = ts.ckks_tensor(context, x).reshape_([channel,height,width])
#print(f"{np.array(enc_x.decrypt().tolist())}\n")


# convolution parameter
out_channels = 2
kernel_size = 3
stride = 1
padding = (0,0)


print("==> Convolution...")
conv = torch.nn.Conv2d(channel, out_channels, kernel_size, stride, padding, bias=True)
y = conv(x)
print(f"{np.array(y.tolist())}\n")

print("==> Ecrypted Convolution as Matrix Multiplication...")
conv_w = get_weight("conv", conv)
enc_y = conv2d_as_matrix_mul(enc_x, conv_w)
print(f"{np.array(enc_y.decrypt().tolist())}\n")


'''
print("==> plain normal convolution...")
conv = torch.nn.Conv2d(channel, out_channels, kernel_size, stride, padding)
y = conv(x)
print(f"{np.array(y.tolist())}\n")

print(f"==> encrypted normal convolution...")
conv_para = get_conv_para(conv)

t1 = time.time()
enc_y = conv2d(enc_x, conv_para)
t2 = time.time()
print(f"{t2-t1} seconds\n")
print(f"{np.array(enc_y.decrypt().tolist())}\n")
'''


'''
print("==> plain depthwise convolution...")
conv = torch.nn.Conv2d(channel, channel, kernel_size, stride, padding, groups=channel)
y = conv(x)
print(f"{np.array(y.tolist())}\n")

print(f"==> encrypted depthwise convolution...")
conv_para = get_conv_dw_para(conv)

t1 = time.time()
enc_y = conv2d_dw(enc_x, conv_para)
t2 = time.time()
print(f"{t2-t1} seconds\n")
print(f"{np.array(enc_y.decrypt().tolist())}\n")
'''