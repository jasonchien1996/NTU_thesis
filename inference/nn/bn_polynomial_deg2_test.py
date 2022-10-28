import time
import torch
import numpy as np
import tenseal as ts
from util import *
from bn_polynomial_deg2 import *


np.set_printoptions(6, suppress=True, linewidth=120)


print("==> prepare context...")
context = get_context(2,8192,30)

# generate data
print("==> prepare data...")
channel = 1
height = 5
width = 5
# plain data
x = torch.from_numpy(np.random.random((1,channel,height,width))).float()
#print(f"{np.array(x.tolist())}\n")
# encrypted data
enc_x = ts.ckks_tensor(context, x).reshape_([channel,height,width])
#print(f"{np.array(enc_x.decrypt().tolist())}\n")


print("==> plain batch normalization...") 
train_data = torch.from_numpy(np.random.random((32,channel,height,width))).float()
bn = torch.nn.BatchNorm2d(channel)
bn.training=True
bn.track_running_stats=True

bn(train_data)

bn.training=False
bn.track_running_stats=True

bn(x)

print("==> plain polynomial activation...")
a, b, c = 0.5, 0.4, 0.3
y = a + b*x + c*x*x

print(f"{np.array(y.tolist())}\n")

print(f"==> bn_polynomial_deg2...")
bn_para = get_bn_para(bn)

t1 = time.time()
enc_y = bn_polynomial_deg2(enc_x, bn_para, [a, b, c], False)
t2 = time.time()
print(f"{t2-t1} seconds\n")
print(f"{np.array(enc_y.decrypt().tolist())}")