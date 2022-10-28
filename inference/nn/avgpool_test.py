import torch
import time
import numpy as np
import tenseal as ts
from test_util import *
from avgpool import *

np.set_printoptions(6, suppress=True, linewidth=120)


print("==> prepare context...")
context = get_context(1,8192,40)

# generate data
print("==> prepare data...")
channel = 3
height = 10
width = 10
# plain data
x = torch.from_numpy(np.random.random((1,channel,height,width))).float()
# encrypted data
enc_x = ts.ckks_tensor(context, x).reshape_([channel,height,width])
#print(x)

print("==> plain avgpool...") 
kernel_size = 2
stride = 1
padding = (0,0)
divisor = 4
pool = torch.nn.AvgPool2d(kernel_size, stride, padding, divisor_override=divisor)
y = pool(x)
y = y.view(y.size(0), -1)
print(f"{np.array(y.tolist())}\n")


print(f"==> encrypted avgpool...")
w = get_weight("pool", pool)
m, _ = get_avgpool2d_matrix((channel,height,width),w)
#print(m)
position, kernel = matrix_to_filter((channel,height,width), m)
#print(kernel[0].shape())
enc_x.pool_(position, kernel).mul_(0.25)
print(f"{np.array(enc_x.decrypt().tolist())}")