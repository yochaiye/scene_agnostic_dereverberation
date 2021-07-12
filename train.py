#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:33:56 2020

@author: yochai_yemini
"""

import torch
from torch.utils.data import DataLoader
from networks.model_dss import weights_init
import losses
import importlib
import random
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import numpy as np

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pathlib
import logging
from barbar import Bar
import argparse
from datetime import datetime
import wandb
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test_step(net, dataloader, criterion, device):
    net.eval()
    avg_loss = 0
    with torch.no_grad():
        for (Z, S) in dataloader:
            Z = Z.to(device)
            S = S.to(device)

            pred = net(Z)
            loss = criterion(pred, S)
            avg_loss += loss.item()
    avg_loss = avg_loss / len(dataloader)
    net.train()
    return avg_loss


# Args parser
parser = argparse.ArgumentParser('')

parser.add_argument('--mics_num', help='number of microphones in the array.', type=int, default=2)
parser.add_argument('--batch_size', help='mini-batch size.', type=int, default=64)
parser.add_argument('--df', help='downsampling factor to get less training samples.', type=int, default=2)
parser.add_argument('--dataset', help='BIUREV/BIUREV-N', type=str, default='BIUREV', choices=['BIUREV', 'BIUREV-N'])
parser.add_argument('--epochs_num', help='number of training epochs.', type=int, default=70)
parser.add_argument('--lr', help='initial learning rate.', type=float, default=3e-4)
parser.add_argument('--ngf', help='base number of filters.', type=int, default=64)
parser.add_argument('--kernel_type', help='size of kernels in the network', type=str, default='square', choices=['square', 'rect_vert', 'rect_horiz'])
parser.add_argument('--unet_arch', help='U-net architecture.', type=str, default='vanilla', choices=['dss', 'vanilla'])
parser.add_argument('--loss_func', help='desired loss function.', type=str, default='grad_mse', choices=['mse', 'l1', 'grad_mse', 'grad_l1'])

# gpu choice
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU', type=str, default='3')

# reproducibility
parser.add_argument('--seed', help='choose seed for randomness.', type=int, default=0)

# Weights and Biases Interface
parser.add_argument('--use_wandb', help='use w&b interface', type=str2bool, default=False)

args = parser.parse_args()

# set gpus
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
ngpu = len(gpu_ids)
device = torch.device("cuda:{}".format(gpu_ids[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Set seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# create a results file to save checkpoints
now = datetime.now()  # datetime object containing current date and time
dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
train_dir = pathlib.Path('./trained_models/' + 'mics{}_'.format(args.mics_num) + dt_string)
train_dir.mkdir(parents=True, exist_ok=True)

# save the experiment's setting to a file
file_name = train_dir / 'settings.txt'
args_dict = vars(args)
with open(file_name, 'wt') as opt_file:
    for k, v in sorted(args_dict.items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))

# logging
log_file = train_dir / "training.log"
logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())  # write to log and print to screen
logging.info("Random Seed: %d" % args.seed)
if args.use_wandb:
    wandb.init(project='dereverberation_with_sets')
    wandb.config.update(args)

# Create a folder for trained models if it doesn't exist
models_dir = train_dir / 'checkpoints'
models_dir.mkdir(parents=True, exist_ok=True)

# Data loader
from data import reverb_dataset as dataset
trainloader = DataLoader(dataset.ReverbDataset(args.mics_num, args.dataset, 'train', args.df),
                         args.batch_size,
                         shuffle=True,
                         num_workers=4)
valnearloader = DataLoader(dataset.ReverbDataset(args.mics_num, args.dataset, 'val_near', 1),
                           args.batch_size,
                           num_workers=4)
valfarloader = DataLoader(dataset.ReverbDataset(args.mics_num, args.dataset, 'val_far', 1),
                          args.batch_size,
                          num_workers=4)

# Model
kernel_size = {'square': 4, 'rect_vert': (8, 4), 'rect_horiz': (4, 8)}
if args.unet_arch == 'dss':
    from networks.model_dss import UNet
    net = UNet(args.ngf, 1, kernel_size[args.kernel_type]).to(device)
elif args.unet_arch == 'vanilla':
    from networks.model import UNet
    net = UNet(args.ngf, args.mics_num, kernel_size[args.kernel_type]).to(device)
num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
logging.info('Number of trainable parameters is {}'.format(num_trainable_params))
if args.use_wandb:
    wandb.watch([net])

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    net = nn.DataParallel(net, gpu_ids)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
net.apply(weights_init)

# Setup an Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr)
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                factor=0.5, patience=6,
                                                threshold=2e-4,
                                                threshold_mode='abs',
                                                verbose=True)

# Training Loop
logging.info("Starting Training Loop...")
loss_funcs = {'mse': nn.MSELoss(),
              'l1': nn.L1Loss(),
              'grad_l1': losses.GradL1Loss(device),
              'grad_mse': losses.GradMSELoss(device),
              }
criterion = loss_funcs[args.loss_func]
train_loss = []
val_near_loss = []
val_far_loss = []
for epoch in range(args.epochs_num):
    train_loss_epoch = 0

    # Train step
    for (Z, S) in Bar(trainloader):
        net.zero_grad()

        S = S.to(device)
        Z = Z.to(device)

        pred = net(Z)
        loss = criterion(pred, S)
        train_loss_epoch += loss.item()

        loss.backward()
        optimizer.step()

    train_loss_epoch = train_loss_epoch / len(trainloader)
    train_loss.append(train_loss_epoch)

    # Test step
    # Calculate val loss for near condition
    val_near_loss_epoch = test_step(net, valnearloader, criterion, device)
    val_near_loss.append(val_near_loss_epoch)

    # Calculate val loss for far condition
    val_far_loss_epoch = test_step(net, valfarloader, criterion, device)
    val_far_loss.append(val_far_loss_epoch)

    lr_sched.step(val_far_loss_epoch + val_far_loss_epoch)

    logging.info('[%d/%d]: train_loss=%.4f, val_near_loss=%.4f, val_far_loss=%.4f' %
                 (epoch, args.epochs_num, train_loss_epoch, val_near_loss_epoch,
                  val_far_loss_epoch))
    if args.use_wandb:
        wandb.log({'train_loss': train_loss, 'val_near_loss': val_near_loss, 'val_far_loss': val_far_loss})

    # Save model
    if epoch == (args.epochs_num - 1):
        # save model
        filename = models_dir / 'epoch{}.pt'.format(epoch)
        torch.save(net.state_dict(), filename)
        if args.use_wandb:
            torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'epoch%d.pt' % epoch))

# plt.figure()
# plt.plot(train_loss, label='train loss')
# plt.plot(val_near_loss, label='val loss far')
# plt.plot(val_far_loss, label='val loss near')
# plt.legend()
# plt.grid(True)
# plt.savefig(train_dir / 'losses', bbox_inches='tight')
#