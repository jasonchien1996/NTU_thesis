import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class approxRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.117071*x*x + 0.5*x + 0.375373

class Deep(nn.Module):
    def __init__(self, params):
        super(Deep, self).__init__()
        width_multiplier = params.width_multiplier

        self.conv1 = nn.Conv2d(3, 4*width_multiplier, kernel_size=5, stride=1, padding=(2,2), bias=True)
        self.bn1 = nn.BatchNorm2d(4*width_multiplier)
        
        self.act1 = nn.ReLU()
        self.act1 = approxRELU()
        
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)
        
        self.conv3 = nn.Conv2d(4*width_multiplier, 8*width_multiplier, kernel_size=3, stride=2, padding=(1,1), bias=True)  
        self.conv4 = nn.Conv2d(8*width_multiplier, 16*width_multiplier, kernel_size=3, stride=2, padding=(1,1), bias=True)
        self.bn3 = nn.BatchNorm2d(16*width_multiplier)
        '''
        self.conv3 = nn.Conv2d(4*width_multiplier, 16*width_multiplier, kernel_size=7, stride=4, padding=(3,3), bias=True)
        self.bn3 = nn.BatchNorm2d(16*width_multiplier)
        '''
        self.act2 = nn.ReLU()
        self.act2 = approxRELU()        
        
        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=(1,1), divisor_override=9)
        self.linear1 = nn.Linear(1024*params.width_multiplier, 10)
        '''
        self.linear1 = nn.Linear(1024*params.width_multiplier, 256*params.width_multiplier)
        self.bn5 = nn.BatchNorm1d(256*params.width_multiplier)

        self.act3 = nn.ReLU()
        self.act3 = approxRELU() 
        
        self.dropout = torch.nn.Dropout(params.dropout_rate)
        self.linear3 = nn.Linear(256*params.width_multiplier, 128*params.width_multiplier)
        self.bn7 = nn.BatchNorm1d(128*params.width_multiplier)
        
        self.act4 = nn.ReLU()
        self.act4 = approxRELU() 
        
        self.linear4 = nn.Linear(128*params.width_multiplier, 10)
        '''
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        
        x = self.act1(x)
        x = self.avg1(x)

        #x = self.bn3(self.conv3(x))
        x = self.bn3(self.conv4(self.conv3(x)))
        
        x = self.act2(x)
        x = self.avg2(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        '''
        x = self.bn5(self.linear1(x))
        
        x = self.act3(x)
        
        x = self.dropout(x)
        x = self.bn7(self.linear3(x))
        
        x = self.act4(x)
        
        x = self.linear4(x)
        '''
        
        return x
        
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