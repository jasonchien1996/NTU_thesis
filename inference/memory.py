# profiling the memory consumption of encrypting a cifar10 image
import sys
import torch
import tenseal as ts
import torchvision.transforms as transforms
from torchvision import datasets
from memory_profiler import profile

torch.manual_seed(73)

transform_test = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

testset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1)

# get the first image
img = None
for data, target in testloader:
    img = data
    break


level = 8
bits_scale = 23
modulus_degree = 32768

@profile
def memory_footprint():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=modulus_degree, coeff_mod_bit_sizes = [bits_scale+5]+[bits_scale]*level+[bits_scale+5])
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()
    
    enc_input = ts.ckks_tensor(context, img)
        
memory_footprint()