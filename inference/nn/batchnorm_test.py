import torch
import numpy as np
import tenseal as ts
from util import *
from batchnorm import *

np.set_printoptions(6, suppress=True, linewidth=120)


print("==> prepare context...")
context = get_context(1,8192,30)

# generate data
print("==> prepare data...")
channel = 2
height = 5
width = 5
# plain data
x = torch.from_numpy(np.random.random((1,channel,height,width))).float()
# encrypted data
enc_x = ts.ckks_tensor(context, x).reshape_([channel,height,width])
input_size = np.prod(enc_x.shape)
print(f"{np.array(enc_x.decrypt().tolist())}\n")

'''
print("==> plain BN2d...") 
train_data = torch.from_numpy(np.random.random((32,channel,height,width))).float()
bn = torch.nn.BatchNorm2d(channel)
bn(train_data)

bn.training=False
bn.track_running_stats=True

y = bn(x)
print(f"{np.array(y.tolist())}\n")
'''

'''
print(f"==> encrypted batch normalization...")
bn_para = get_bn_para(bn)
enc_y = batchnorm2d(enc_x, bn_para)
print(f"{np.array(enc_y.decrypt().tolist())}")
'''


'''
print(f"==> encrypted BN2d as matrix multiplication...")
bn_para = get_bn_para(bn)
A,B = get_batchnorm2d_matrix_form((channel,height,width), bn_para)
enc_x.reshape_([1,np.prod(enc_x.shape)]).mm_(A).add_(B).reshape_([channel, height, width])
print(f"{np.array(enc_x.decrypt().tolist())}")
'''

print("==> plain BN1d...") 
train_data = torch.from_numpy(np.random.random((32,input_size))).float()
bn = torch.nn.BatchNorm1d(input_size)
bn(train_data)

bn.training=False
bn.track_running_stats=True

y = bn(x.view(x.size(0), -1))
print(f"{np.array(y.tolist())}\n")


print(f"==> encrypted BN1d as matrix multiplication...")
bn_para = get_bn_para(bn)
A,B = get_batchnorm1d_matrix_form(input_size, bn_para)
enc_x.reshape_([1,input_size]).mm_(A).add_(B)
print(f"{np.array(enc_x.decrypt().tolist())}")