'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
import logging
import numpy as np

def get_time():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return current_time

def mkdir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def mk_log_dir(args):
    mkdir(args.tensorboard)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_name = args.model + '_lr' + str(args.lr) + '_' + args.comment + current_time
    log_dir = os.path.join(args.tensorboard, log_name)
    return log_name, log_dir

def mk_monitor_dir(args):
    mkdir(args.tensorboard)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_name = args.content + '_'  + current_time
    log_dir = os.path.join(args.tensorboard, log_name)
    return log_dir

class trainStrategy(object):
    # Deep Residual Learning for Image Recognition
    def __init__(self) -> None:
        super().__init__()
        self.wd = 1e-4
        self.init_lr = 1e-1
        self.schedule_rate = 1e-1
        self.schedule_point = [90, 140]
        self.epoch = 180
    
    def modify():
        print("not yet")

class txt_logger(object):

    def __init__(self, save_dir, name, filename):

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        self.logger = logger

        self.info = {}

    def add_scalar(self, tag, value, step=None, iter=None):
        if tag in self.info:
            self.info[tag].append(value)
        else:
            self.info[tag] = [value]


    def print_info(self, epoch):

        info_line = 'epoch:{},'.format(epoch)
        for i in self.info.keys():
            info = np.array(self.info[i]).mean()
            info_line += i + ':' + str(round(info,4)) + ', '

        print(info_line)
        self.logger.info(
            info_line
        )
        self.info = {}
    
    def print_info_iter(self, epoch, iter):

        info_line = 'epoch:{}, iter:{},'.format(epoch,iter)
        for i in self.info.keys():
            info = np.array(self.info[i]).mean()
            info_line += i + ':' + str(round(info,4)) + ', '

        print(info_line)
        self.logger.info(
            info_line
        )
        self.info = {}

def load_checkpoint(net):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return best_acc, start_epoch

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




def progress_bar(current, total, msg=None):

    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    # global last_time, begin_time
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
