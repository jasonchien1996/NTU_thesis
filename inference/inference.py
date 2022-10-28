import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models import shallow, shallow_ext, shallow_ENC, shallow_ENC_M, shallow_ext_ENC
from util import get_context, enc_test
from torchvision import datasets

torch.manual_seed(73)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Load one element at a time
print('==> Preparing data...')
'''
# mnist
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
testset = datasets.MNIST(root='./mnist_', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1)
'''
# cifar10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
testset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1)


# Plain Model
print("\n==> initializing model...\n")
model = shallow(width_multiplier=1)
#model = shallow_ext(width_multiplier=1)

model = model.to(device)
path = './checkpoint/s1.pth.tar'
#path = './checkpoint/s1_ext.pth.tar'
checkpoint = torch.load(path, map_location=lambda storage, loc: storage) #CPU
#checkpoint = torch.load(path) # GPU

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model.load_state_dict(checkpoint['state_dict'])

print("==> fusing model...")
t1 = time.time()
enc_model = shallow_ENC(model)
#enc_model = shallow_ENC_M(model)
#enc_model = shallow_ext_ENC(model)
t2 = time.time()
print(f"{t2-t1} seconds\n")

# different HE settings
#context = get_context( level=5, poly_mod_degree=8192, bits_scale=21, integer=5 )
#context = get_context( level=5, poly_mod_degree=8192, bits_scale=28, integer=5 )
#context = get_context( level=6, poly_mod_degree=8192, bits_scale=21, integer=5 )
#context = get_context( level=6, poly_mod_degree=8192, bits_scale=26, integer=5 )
#context = get_context( level=9, poly_mod_degree=16384, bits_scale=23, integer=5 )
#context = get_context( level=11, poly_mod_degree=16384, bits_scale=23, integer=5 )
context = get_context( level=11, poly_mod_degree=16384, bits_scale=30, integer=10 )
#context = get_context( level=19, poly_mod_degree=32768, bits_scale=24, integer=5 )

criterion = torch.nn.CrossEntropyLoss()
enc_test(context, enc_model, model, test_loader, criterion)