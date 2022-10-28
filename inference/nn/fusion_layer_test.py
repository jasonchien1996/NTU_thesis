import sys
import time
import torch
import numpy as np
import tenseal as ts
from test_util import *
from linear import *
from avgpool import *
from batchnorm import *
from convolution import *

np.set_printoptions(precision=3, edgeitems=7, suppress=True, linewidth=180)

print("==> prepare context...")
context = get_context(level=2,poly_mod_degree=8192,bits_scale=30,integer=15)

# generate data
print("==> prepare data...\n")
channel = 1
height = 10
width = 10
data = np.random.random((channel,height,width))

# encrypted data
enc_x = ts.ckks_tensor(context, data)
#print(f"{np.array(enc_x.decrypt().tolist())}\n")

# plain data
x = torch.from_numpy(data.reshape((1,channel,height,width))).float()
#print(f"{np.array(x.tolist())}\n")


# avgpool1
kernel_size = 3
stride = 1
padding = (1,1)
divisor = 9
print("==> plain avgpool...") 
pool1 = torch.nn.AvgPool2d(kernel_size, stride, padding, divisor_override=divisor)
x = pool1(x)
out_size1= x.shape[1:]
#print(f"{np.array(x.tolist())}\n")

# convolution1
out_channels = 1
kernel_size = 3
stride = 1
padding = (1,1)
print("==> plain convolution...")
conv1 = torch.nn.Conv2d(channel, out_channels, kernel_size, stride, padding)
x = conv1(x)
out_size2 = x.shape[1:]
#print(f"{np.array(x.tolist())}\n")

# batchnorm1
print("==> plain batch normalization...")
train_data = torch.from_numpy(np.random.random((32,out_size2[0],out_size2[1],out_size2[2]))).float()
bn1 = torch.nn.BatchNorm2d(out_channels)
bn1(train_data)

bn1.training=False
bn1.track_running_stats=True

x = bn1(x)
out_size3 = x.shape[1:]
print(f"{np.array(x.tolist())}\n")


# merge
pool_w1 = get_weight("pool", pool1)
conv_w1 = get_weight("conv", conv1)
bn_w1 = get_weight("bn", bn1)

pool1 = get_avgpool2d_matrix((channel,height,width), pool_w1)
conv1 = get_conv2d_matrix(out_size1, conv_w1)
bn1 = get_bn2d_matrix(out_size2, bn_w1)


m1, b1 = merge([pool1, conv1, bn1])

position_list1, kernel_list1 = matrix_to_filter((channel,height,width), m1)
print(len(position_list1))
print(len(kernel_list1))

print("==> encrypted merged layer...")
t1 = time.time()
enc_x.conv_fusion_(position_list1, kernel_list1)
enc_x.add_(b1)
enc_x.reshape_(out_size3)
t2 = time.time()
print(f"{t2-t1} seconds")
print(f"{np.array(enc_x.decrypt().tolist())}\n")
