import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tenseal as ts
from nn import *
from util import *

class approxRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.117071*x*x + 0.5*x + 0.375373
        

class shallow(nn.Module):
    def __init__(self, width_multiplier=1):
        super(shallow, self).__init__()
        
        self.width_multiplier = width_multiplier

        self.conv1 = nn.Conv2d(3, 4*width_multiplier, kernel_size=5, stride=1, padding=(2,2), bias=True)
        self.bn1 = nn.BatchNorm2d(4*width_multiplier)
        
        self.act1 = approxRELU()
        
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)

        self.conv3 = nn.Conv2d(4*width_multiplier, 16*width_multiplier, kernel_size=7, stride=4, padding=(3,3), bias=True)
        self.bn3 = nn.BatchNorm2d(16*width_multiplier)
        
        self.act2 = approxRELU()        
        
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)
        self.linear1 = nn.Linear(1024*width_multiplier, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.avg1(x)
        x = self.bn3(self.conv3(x))
        x = self.act2(x)
        x = self.avg2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


class shallow_ext(nn.Module):
    def __init__(self, width_multiplier=1):
        super(shallow_ext, self).__init__()
        
        self.width_multiplier = width_multiplier

        self.conv1 = nn.Conv2d(3, 4*width_multiplier, kernel_size=5, stride=1, padding=(2,2), bias=True)
        self.bn1 = nn.BatchNorm2d(4*width_multiplier)
        
        self.act1 = approxRELU()
        
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)
        
        self.conv3 = nn.Conv2d(4*width_multiplier, 8*width_multiplier, kernel_size=3, stride=2, padding=(1,1), bias=True)        
        self.conv4 = nn.Conv2d(8*width_multiplier, 16*width_multiplier, kernel_size=3, stride=2, padding=(1,1), bias=True)
        self.bn3 = nn.BatchNorm2d(16*width_multiplier)
        
        self.act2 = approxRELU()        
        
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)
        self.linear1 = nn.Linear(1024*width_multiplier, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.avg1(x)
        x = self.bn3(self.conv4(self.conv3(x)))
        x = self.act2(x)
        x = self.avg2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x
        
class shallow_ENC():
    def __init__(self, model):
        self.width_multiplier = model.width_multiplier
    
        w = get_weight("conv", model.conv1)
        self.conv1_m = get_matrix("conv", w, (3,32,32))
        print(self.conv1_m[0].shape)
        
        self.bn1 = get_weight("bn2d", model.bn1)
        bn1_m = get_matrix("bn2d", self.bn1, (4*self.width_multiplier,32,32))
        print(bn1_m[0].shape)
        
        w = get_weight("pool", model.avg1)
        avg1_m = get_matrix("pool", w, (4*self.width_multiplier,32,32))
        print(avg1_m[0].shape)
        
        w = get_weight("conv", model.conv3)
        self.conv3_m = get_matrix("conv", w, (4*self.width_multiplier,32,32))
        print(self.conv3_m[0].shape)
        
        self.bn3 = get_weight("bn2d", model.bn3)
        bn3_m = get_matrix("bn2d", self.bn3, (16*self.width_multiplier,8,8))
        print(bn3_m[0].shape)
        
        w = get_weight("pool", model.avg2)
        avg2_m = get_matrix("pool", w, (16*self.width_multiplier,8,8))
        print(avg2_m[0].shape)
        
        self.linear = get_weight("linear", model.linear1)
        linear_m = get_matrix("linear", self.linear)
        print(linear_m[0].shape)
        
        self.pos1, self.k1 = matrix_to_filter((3,32,32), self.conv1_m[0])
        self.pos2, self.k2 = matrix_to_filter((4*self.width_multiplier,32,32), avg1_m[0])
        self.pos3, self.k3 = matrix_to_filter((4*self.width_multiplier,32,32), self.conv3_m[0])
        self.pos4, self.k4 = matrix_to_filter((16*self.width_multiplier,8,8), avg2_m[0])

    def forward(self, enc_x):
        a = 0.117071
        b = 0.5
        c = 0.375373
        coefficient = [c, b, a]
        
        enc_x.reshape_([3,32,32])

        # conv layer 1
        print("==> conv layer 1...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos1, self.k1).add_(self.conv1_m[1])
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")

        # bn layer 1
        print("==> bn layer 1...")
        t1 = time.time()
        enc_x = batchnorm2d(enc_x, self.bn1)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # activation 1
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # pool layer 1
        print("==> pool layer 1...")
        t1 = time.time()
        enc_x.pool_(self.pos2, self.k2).mul_(0.11111)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # conv layer 2
        print("==> conv layer 2...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos3, self.k3).add_(self.conv3_m[1])
        t2 = time.time()
        enc_x.reshape_([16*self.width_multiplier,8,8])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # bn layer 2
        print("==> bn layer 2...")
        t1 = time.time()
        enc_x = batchnorm2d(enc_x, self.bn3)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # activation 2
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # pool layer 2
        print("==> pool layer 2...")
        t1 = time.time()
        enc_x.pool_(self.pos4, self.k4).mul_(0.11111)
        t2 = time.time()
        enc_x.reshape_([16*self.width_multiplier,8,8])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # linear layer
        print("==> linear layer...")
        t1 = time.time()
        enc_x = linear(enc_x, self.linear)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        return enc_x
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        

class shallow_ENC_M():
    def __init__(self, model):
        self.width_multiplier = model.width_multiplier
    
        w = get_weight("conv", model.conv1)
        conv1_m = get_matrix("conv", w, (3,32,32))
        print(conv1_m[0].shape)
        
        w = get_weight("bn2d", model.bn1)
        bn1_m = get_matrix("bn2d", w, (4*self.width_multiplier,32,32))
        print(bn1_m[0].shape)
        
        w = get_weight("pool", model.avg1)
        avg1_m = get_matrix("pool", w, (4*self.width_multiplier,32,32))
        print(avg1_m[0].shape)
        
        w = get_weight("conv", model.conv3)
        m, b = get_matrix("conv", w, (4*self.width_multiplier,32,32))
        conv3_m = (m*(0.117071/9),b)
        print(conv3_m[0].shape)
        
        w = get_weight("bn2d", model.bn3)
        bn3_m = get_matrix("bn2d", w, (16*self.width_multiplier,8,8))
        print(bn3_m[0].shape)
        
        w = get_weight("pool", model.avg2)
        avg2_m = get_matrix("pool", w, (16*self.width_multiplier,8,8))
        print(avg2_m[0].shape)
        
        w = get_weight("linear", model.linear1)
        m, b = get_matrix("linear", w)
        linear_m = (m*0.117071,b)
        print(linear_m[0].shape)
        
        # m.shape = ( _ , flatten input size)
        # b.shape = ( _ )
        m1, self.b1 = merge([conv1_m, bn1_m])
        m2, self.b2 = merge([conv3_m, bn3_m])
        # m2, self.b2 = merge([avg1_m, conv3_m, bn3_m])
        self.m3, self.b3 = merge([avg2_m, linear_m])
        
        self.pos1, self.k1 = matrix_to_filter((3,32,32), m1)
        self.pos_avg, self.k_avg = matrix_to_filter((4,32,32), avg1_m[0])
        self.pos2, self.k2 = matrix_to_filter((4*self.width_multiplier,32,32), m2)
        
    def forward(self, enc_x):
        a = 1
        b = 0.5/0.117071
        c = 0.375373/0.117071
        coefficient = [c, b, a]
        
        enc_x.reshape_([3,32,32])

        # fusion layer 1
        print("==> fusion layer 1...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos1, self.k1).add_(self.b1)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")

        # activation 1
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # pool layer 1
        print("==> pool layer 1...")
        t1 = time.time()
        enc_x.pool_(self.pos_avg, self.k_avg)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # fused layer 2
        print("==> fusion layer 2...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos2, self.k2).add_(self.b2)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # activation 2
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # fusion layer 3
        print("==> fusion layer 3...")
        t1 = time.time()
        enc_x.reshape_([1,np.prod(enc_x.shape)])
        enc_x.mm_(self.m3.T)
        if self.b3 is not None:
            enc_x.add_(self.b3)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        return enc_x
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        

class shallow_ext_ENC():
    def __init__(self, model):
        self.width_multiplier = model.width_multiplier
    
        w = get_weight("conv", model.conv1)
        conv1_m = get_matrix("conv", w, (3,32,32))
        print(conv1_m[0].shape)
        
        w = get_weight("bn2d", model.bn1)
        bn1_m = get_matrix("bn2d", w, (4*self.width_multiplier,32,32))
        print(bn1_m[0].shape)
        
        w = get_weight("pool", model.avg1)
        avg1_m = get_matrix("pool", w, (4*self.width_multiplier,32,32))
        print(avg1_m[0].shape)
        
        w = get_weight("conv", model.conv3)
        m, b = get_matrix("conv", w, (4*self.width_multiplier,32,32))
        self.conv3_m = (m*(0.117071/9),b)
        print(self.conv3_m[0].shape)
        
        w = get_weight("conv", model.conv4)
        conv4_m = get_matrix("conv", w, (8*self.width_multiplier,16,16))
        print(conv4_m[0].shape)
        
        w = get_weight("bn2d", model.bn3)
        bn3_m = get_matrix("bn2d", w, (16*self.width_multiplier,8,8))
        print(bn3_m[0].shape)
        
        w = get_weight("pool", model.avg2)
        avg2_m = get_matrix("pool", w, (16*self.width_multiplier,8,8))
        print(avg2_m[0].shape)
        
        w = get_weight("linear", model.linear1)
        m, b = get_matrix("linear", w)
        linear_m = (m*0.117071,b)
        print(linear_m[0].shape)
        
        # m.shape = ( _ , flatten input size)
        # b.shape = ( _ )
        m1, self.b1 = merge([conv1_m, bn1_m])
        self.pos1, self.k1 = matrix_to_filter((3,32,32), m1)
        
        self.pos2, self.k2 = matrix_to_filter((4,32,32), avg1_m[0])
        
        self.pos3, self.k3 = matrix_to_filter((4*self.width_multiplier,32,32), self.conv3_m[0])
        
        m2, self.b2 = merge([conv4_m, bn3_m])
        self.pos4, self.k4 = matrix_to_filter((8*self.width_multiplier,16,16), m2)
        
        self.m3, self.b3 = merge([avg2_m, linear_m])
        
    def forward(self, enc_x):
        a = 1
        b = 0.5/0.117071
        c = 0.375373/0.117071
        coefficient = [c, b, a]
        
        enc_x.reshape_([3,32,32])

        # fusion layer 1
        print("==> fusion layer 1...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos1, self.k1).add_(self.b1)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")

        # activation 1
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # pool layer 1
        print("==> pool layer 1...")
        t1 = time.time()
        enc_x.pool_(self.pos2, self.k2)
        t2 = time.time()
        enc_x.reshape_([4*self.width_multiplier,32,32])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # conv layer 2
        print("==> conv layer 2...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos3, self.k3).add_(self.conv3_m[1])
        t2 = time.time()
        enc_x.reshape_([8*self.width_multiplier,16,16])
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
         
        # fusion layer 2
        print("==> fusion layer 2...")
        t1 = time.time()
        enc_x.conv_fusion_(self.pos4, self.k4).add_(self.b2)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # activation 2
        print("==> polynomial activation...")
        t1 = time.time()
        enc_x.polyval_(coefficient)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        # fusion layer 3
        print("==> fusion layer 3...")
        t1 = time.time()
        enc_x.reshape_([1,np.prod(enc_x.shape)])
        enc_x.mm_(self.m3.T)
        if self.b3 is not None:
            enc_x.add_(self.b3)
        t2 = time.time()
        print(f"shape {enc_x.shape}, {t2-t1} seconds\n")
        
        return enc_x
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
          
def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
    
def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}