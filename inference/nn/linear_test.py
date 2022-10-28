import sys
import time
import torch
import numpy as np
import tenseal as ts
from linear import *
from test_util import *

np.set_printoptions(6, suppress=True, linewidth=120)


print("==> prepare context...")
context = get_context(1,8192,30)

# generate data
print("==> prepare data...")
channel = 1
height = 5
width = 5
# plain data
x = torch.from_numpy(np.random.random((1,channel,height,width))).float()
x = x.view(x.size(0), -1)

# encrypted data
enc_x = ts.ckks_tensor(context, x).reshape_([channel,height,width])
print(f"{np.array(enc_x.decrypt().tolist())}\n")

print("==> plain linear...")
in_features = x.shape[-1]
out_features = 10
fc = torch.nn.Linear(in_features, out_features)
y = fc(x)
print(f"{np.array(y.tolist())}\n")

print(f"==> encrypted linear...")
linear_w = get_weight("linear", fc)

t1 = time.time()
enc_y = linear(enc_x, linear_w)
t2 = time.time()
print(f"{t2-t1} seconds\n")
print(f"{np.array(enc_y.decrypt().tolist())}")