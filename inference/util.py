import os
import sys
import time
import math
import torch
import numpy as np
import tenseal as ts
import torch.nn as nn
import torch.nn.init as init
from nn import *

# create TenSEAL context
def get_context( level, poly_mod_degree=8192, bits_scale=25, integer=5 ):
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_mod_degree,
    coeff_mod_bit_sizes=[bits_scale+integer] + [bits_scale]*level + [bits_scale+integer])
    
    # set the scale
    context.global_scale = pow(2, bits_scale)
    
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context
    
# extract weights from a layer
def get_weight(layer_type, layer):
    if layer_type == "conv":
        out_channels, in_channels = layer.out_channels, layer.in_channels
        kernel_h, kernel_w = layer.kernel_size[0], layer.kernel_size[1]
        # must be 4 channel
        conv_weight = layer.weight.view( out_channels, in_channels, kernel_h, kernel_w ).tolist()
        conv_bias = None if layer.bias is None else layer.bias.tolist()
        conv_stride = layer.stride[0]
        conv_padding = layer.padding if type(layer.padding) == tuple else (layer.padding, layer.padding)
        return [conv_weight, conv_bias, conv_stride, conv_padding]
        
    elif layer_type == "bn" or layer_type == "bn1d" or layer_type == "bn2d":
        bn_gamma = layer.weight.tolist()
        bn_bias = layer.bias.tolist()
        bn_mean = layer.running_mean.tolist()
        bn_var = layer.running_var.tolist()
        bn_eps = layer.eps
        return [bn_gamma, bn_bias, bn_mean, bn_var, bn_eps]
    
    elif layer_type == "pool":
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding if type(layer.padding) == tuple else (layer.padding, layer.padding)
        divisor = layer.divisor_override 
        return [kernel_size, stride, padding, divisor]
    
    elif  layer_type == "linear":
        fc_weight = layer.weight.T.tolist()
        fc_bias = layer.bias.tolist()
        return [fc_weight, fc_bias]
        
    else:
        print(f"Error: no layer type {layer_type}.")
        sys.exit()

# get the representation of the weights of a layer
def get_matrix(layer_type, weights, input_size=None):
    if layer_type == "conv":
        matrix = get_conv2d_matrix(input_size, weights)
    
    elif layer_type == "bn1d":
        matrix = get_bn1d_matrix(input_size, weights)
            
    elif layer_type == "bn2d":
        matrix = get_bn2d_matrix(input_size, weights)
        
    elif layer_type == "pool":
        matrix = get_avgpool2d_matrix(input_size, weights)
    
    elif layer_type == "linear":
        matrix = get_linear_matrix(weights)
        
    else:
        print(f"Error: no layer type {layer_type}.")
        sys.exit()
    # matrix = (weight matrix, bias)
    return matrix
        
# use matrix multiplication to merge layers
def merge(layer_list):
    A, B = layer_list[0]
    for M, N in layer_list[1:]:
        A = np.matmul(M, A)
        if B is None:
            B = N
        else:
            B = np.matmul(M, B)
            if N is not None:
                B = B + N
    return (A, B)
    
# call this method before using the new API - pool_, conv_fusion_
def matrix_to_filter(input_shape, matrix):
    """
    Args:
        input_shape    : (C, H, W)
        matrix         : shape = (H_m , C*H*W)
    """
    c_i, h_i, w_i = input_shape[0], input_shape[1], input_shape[2]
    h_m, w_m = matrix.shape[0], matrix.shape[1]
    
    channel_size = h_i*w_i
    if (c_i*channel_size) != w_m:
        print("Error: input_shape does not match the matrix shape!")
        sys.exit()
        
    kernel_list = []
    position_list = []
    for row in range(h_m):   
        m = matrix[row].reshape(input_shape)
        t, l = 0, 0
        d, r = h_i-1, w_i-1
        
        # find the first non-zero element index from the top
        while not np.any(m[:,t,:]):
            t = t + 1
        while not np.any(m[:,d,:]):
            d = d - 1
        while not np.any(m[:,:,l]):
            l = l + 1
        while not np.any(m[:,:,r]):
            r = r - 1
        
        if l > r or t > d:
            print("Error: empty kernel")
            sys.exit()
        
        kernel = m[:,t:d+1,l:r+1]
        
        found = False
        for idx, k in enumerate(kernel_list):
            if np.array_equal(kernel, k):
                found = True
                position_list.append((idx, t, l))
                break
                
        if not found:
            position_list.append((len(kernel_list), t, l))
            kernel_list.append(kernel)
            
    kernel_list = [ts.plain_tensor(k).data for k in kernel_list]      

    return position_list, kernel_list


def enc_test(context, enc_model, model, testloader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    
    for idx, (data, target) in enumerate(testloader):
        # Plain Inference
        print(f"==> Plain inferencing...\n{np.array(model(data.view(1,3,32,32)).tolist())}\n")
        
        # Encoding and Encryption
        start = time.time()
        enc_input = ts.ckks_tensor(context, data)
        end = time.time()
        print(f"==> Encrypting...\n{end-start} seconds\n")
        
        # Encrypted Evaluation
        print(f"==> Encrypted inferencing...")
        start = time.time()
        enc_output = enc_model(enc_input)     
        end = time.time()
        print(f"total inferencing time: {end-start} seconds\n")
        
        # Decryption
        start = time.time()
        output = enc_output.decrypt().tolist()
        end = time.time()
        print(f"==> Decrypting... \n{end-start} seconds\n")
        
        output = torch.tensor(output).view(1, -1)
        print(np.array(output.tolist()))
        
        print("\n===================================================================================================================\n")
        
        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()        
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
               
    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
