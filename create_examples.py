#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:04:17 2020

@author: yochai_yemini
"""

import torch
from data.utils import stft, istft, normalize_mc, normalize_log_spec, denormalize_log_spec, remove_module_str
from networks import model_dss, model, model_attn
import pathlib
import pickle
import numpy as np
import scipy.io
import soundfile as sf
from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

MICS_NUM = 8
FRAMES_NUM = 256
K = 512
overlap = 0.75

# Synthesis window for the ISTFT
MAT = scipy.io.loadmat('synt_win.mat')
synt_win = MAT['synt_win']

# trained_model_path = "./trained_models/mics8_24.03.2021_05:33:29/checkpoints/epoch69.pt"
trained_model_path = "./trained_models/mics8_19.05.2021_01:38:10/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics8_07.05.2021_05:11:49/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics1_02.03.2021_13:06:22/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics8_02.03.2021_12:57:27/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics8_02.03.2021_13:04:41/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics8_02.03.2021_12:59:28/checkpoints/epoch69.pt"
# trained_model_path = "./trained_models/mics8_25.03.2021_04:05:13/checkpoints/epoch69.pt"
spec_type = 'log_mag'      # 'real_imag' 'log_mag'
mics_num = int(trained_model_path[21])
nc = {'log_mag': 1, 'real_imag': 2, 'complex': 1}  # number of input channels

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
if mics_num != 1:
    net = model_dss.UNet(ngf=64, nc=nc[spec_type], kernel_size=4, use_sets=(mics_num != 1), use_extension=True).to(device)
    # net = model_attn.UNet(ngf=32, nc=nc[spec_type], kernel_size=4, use_sets=(mics_num != 1), use_extension=True).to(device)
    # net = model_ori.UNetMC(ngf=64, nc=mics_num, kernel_size=4, use_extension=False).to(device)
else:
    net = model.UNet(ngf=64, nc=nc[spec_type], kernel_size=4, use_extension=False).to(device)
net_state_dict = torch.load(trained_model_path, map_location='cuda:3')
net_state_dict = remove_module_str(net_state_dict)
net.load_state_dict(net_state_dict)

# Get the global min and max values for the spectrograms
# min_max_file = '/mnt/dsi_vol1/users/yochai_yemini/REVERBsim/dataset/mics8/train/global_min_max.p'
min_max_file = '/mnt/dsi_vol1/users/yochai_yemini/REVERBsim_no_noise/dataset/mics8/train/global_min_max.p'
with open(min_max_file, 'rb') as f:
    mag_max_clean, mag_min_clean, mag_max_reverb, mag_min_reverb, \
    real_max_clean, real_max_reverb, real_min_clean, real_min_reverb, \
    imag_max_clean, imag_max_reverb, imag_min_clean, imag_min_reverb =\
        pickle.load(f)
        

def zoomed_subfig(img, ax, x, y, loc, vmin, vmax, extent):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
    axins = zoomed_inset_axes(ax, 2, loc=loc) # zoom = 6
    axins.imshow(img, vmin=vmin, vmax=vmax, aspect='auto', extent=extent)

    # sub region of the original image
    x1, x2 = x # 0.6, 1.2, 0, 0.8
    y1, y2 = y # 0.6, 1.2, 0, 0.8
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0")
    plt.draw()
    # return axins

def save_spectrograms(clean_spec, reverb_spec, enhanced_spec, L):
    fs= 16000
    extent=[0,L/fs,0,8]    
    max_val = 5.3#np.max((mag_max_clean.max(), mag_max_reverb.max()))
    min_val = np.max((mag_min_clean.min(), mag_min_reverb.min()))
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    fig = plt.figure()
    plt.imshow(clean_spec[::-1], vmin=min_val, vmax=max_val, aspect='auto', extent=extent)
    ax = fig.gca()
    plt.colorbar()
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency[KHz]')
    # zoomed_subfig(clean_spec[::-1], ax, [0.6, 1.2], [0, 0.8], 10, min_val, max_val, extent)
    # zoomed_subfig(clean_spec[::-1], ax, [2.5, 2.7], [0, 0.8], 7, min_val, max_val, extent)
    plt.savefig('clean', bbox_inches='tight')
    
    
    
    fig = plt.figure()
    plt.imshow(reverb_spec[::-1], vmin=min_val, vmax=max_val, aspect='auto', extent=[0,L/fs,0,8])
    ax = fig.gca()
    plt.colorbar()
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency[KHz]')
    # zoomed_subfig(reverb_spec[::-1], ax, [0.6, 1.2], [0, 0.8], 10, min_val, max_val, extent)
    # zoomed_subfig(reverb_spec[::-1], ax, [2.5, 2.7], [0, 0.8], 7, min_val, max_val, extent)
    plt.savefig('reverb', bbox_inches='tight')
    
    fig = plt.figure()
    plt.imshow(enhanced_spec[::-1], vmin=min_val, vmax=max_val, aspect='auto', extent=[0,L/fs,0,8])
    ax = fig.gca()
    plt.colorbar()
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency[KHz]')
    # zoomed_subfig(enhanced_spec[::-1], ax, [0.6, 1.2], [0, 0.8], 10, min_val, max_val, extent)
    # zoomed_subfig(enhanced_spec[::-1], ax, [2.5, 2.7], [0, 0.8], 7, min_val, max_val, extent)
    plt.savefig('enhanced', bbox_inches='tight')
    return


def enhance_mag(magnitude, net, device):
    '''
    Enhances the noisy and reverberant log-magnitude 

    Parameters
    ----------
    magnitude : The log-magnitude to be enahnced, dimension are [mics_num x TIME x FREQ]
    net : The trained enhancing network
    device : Device to run on (cpu or gpu)
    '''
    # Normalise to [-1, 1]
    # max_val = np.max(magnitude)
    # min_val = np.min(magnitude)
    # magnitude = utils.normalize_log_spec(magnitude, max_val, min_val)
    magnitude = normalize_log_spec(magnitude, mag_max_reverb, mag_min_reverb)
    # np.random.shuffle(magnitude)

    magnitude = torch.from_numpy(magnitude)
    magnitude = magnitude.type(torch.FloatTensor)
    
    # Divide the input spectrogram into [256, 256] segments to match training
    edge_index = magnitude.shape[1] // FRAMES_NUM   # check how many 256-frames long segments fully fit
    residue = magnitude.shape[1] % FRAMES_NUM
    mics_num = magnitude.shape[0]
    # Concatenate segments along the batch dimension 
    if mics_num == 1:
        to_model = magnitude[:, :edge_index*FRAMES_NUM, :-1].view(-1, 1, FRAMES_NUM, int(K/2))
        if residue != 0:
            last_part = magnitude[:, -FRAMES_NUM:, :-1].view(-1, 1, FRAMES_NUM, int(K/2))
            to_model = torch.cat([to_model, last_part], axis=0)
    else:
        if residue == 0:
            to_model = torch.zeros(edge_index, mics_num, 1, FRAMES_NUM, int(K/2))
        else:
            to_model = torch.zeros(edge_index+1, mics_num, 1, FRAMES_NUM, int(K/2))
        trunc_mag = magnitude[:, :edge_index*FRAMES_NUM, :-1].unsqueeze(1)
        for i in range(edge_index):
            to_model[i] = trunc_mag[:, :, i*FRAMES_NUM:(i+1)*FRAMES_NUM, :]
        if residue != 0:
            to_model[-1] = magnitude[:, -FRAMES_NUM:, :-1].unsqueeze(1)
    
    net.eval()
    with torch.no_grad():
        pred = net(to_model.to(device))
    pred = pred.squeeze().cpu()
    
    # assemble the enhanced magnitude
    if residue == 0:
        recon = pred.view(edge_index*FRAMES_NUM, int(K/2))
    else:
        recon1 = pred[:-1, :].view(edge_index*FRAMES_NUM, int(K/2))
        recon = torch.cat((recon1, pred[-1, -residue:, :]), axis=0)
        
    recon = torch.cat((recon, magnitude[0, :, -1:]), axis=1)    # Add the highest frequency
    
    # Denormalise
    recon = denormalize_log_spec(recon, mag_max_clean, mag_min_clean)
    return recon.numpy()



            
# data_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERBsim/REVERB_WSJCAM0_et/data/')
data_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERBsim_no_noise/REVERB_WSJCAM0_et/data/')

# clean_path = "/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test2/c3u/c3uc020c.wav"
clean_path = "/home/dsi/yochai_yemini/PhD/Year2/RIRgen/clean.wav"
# reverb_path = "/mnt/dsi_vol1/users/yochai_yemini/REVERBsim/REVERB_WSJCAM0_et/data/far_one_near_test/c3q/c3qc020s_ch1.wav"
# reverb_path = "REVERBsim_no_noise/REVERB_WSJCAM0_et/data/random_test/c3u/c3uc020c_ch1.wav"
reverb_path = "/home/dsi/yochai_yemini/PhD/Year2/RIRgen/reverb_ch1.wav"
s, fs = sf.read(clean_path)
s = s/1.1/np.max(np.abs(s))
S, P = stft(s, K, overlap, 'log')

# Get multichannel reverb wavs
z = []
paths = [reverb_path, *[reverb_path[:-5]+str(i)+'.wav' for i in range(2, mics_num+1)]]
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-5_T40c020b.wav',
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-4_T40c020b.wav',
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-3_T40c020b.wav',
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-2_T40c020b.wav',
         #  '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-1_T40c020b.wav',
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-8_T40c020b.wav',
         # '/mnt/dsi_vol1/users/yochai_yemini/REVERB/RealData/MC_WSJ_AV_Eval/audio/stat/T40/array1/5k/AMI_WSJ30-Array1-7_T40c020b.wav']
for j in range(mics_num):   # We have to get all 8 mics for normalisation purpose
    # temp, fs = sf.read(str(reverb_path)[:-5] + '{}.wav'.format(j+1))
    temp, fs = sf.read(paths[j])
    z.append(temp)
    
z = normalize_mc(np.array(z))
z = z[:mics_num]                # take only the desired mics

# for i in range(1, mics_num):
#     temp = np.random.randn(len(z[i]))
#     z[i] = np.std(z[i]) * temp / np.std(temp)

# Get the STFTs of the mutlichannel reverb signals
temp = [stft(z[m], K, overlap, 'log') for m in range(mics_num)]
frames_num = len(temp[0][0].T)
if frames_num < 256:
    raise RuntimeError('Recording is too short.')

Z = {}
Z['mag'] = np.zeros((mics_num, frames_num, int(K/2+1)))
Z['phase'] = np.zeros((mics_num, frames_num, int(K/2+1)))
for i in range(mics_num):
    Z['mag'][i] = temp[i][0].T
    Z['phase'][i] = temp[i][1].T

S_hat = enhance_mag(Z['mag'], net, device)

# Get the phase
closest_mic = np.argmax([np.var(z[i]) for i in range(mics_num)])
z = z[closest_mic]                                    # take only the first microphone for the phase
z = z/1.1/np.max(np.abs(z))
_, recon_phase = stft(z, K, overlap)
s_hat = istft(S_hat.T, recon_phase, synt_win)
frames_num = S.shape[1]
save_spectrograms(S, Z['mag'][0][:frames_num].T, S_hat[:frames_num].T, len(s))
# save_spectrograms(S[:,1000:1400], Z['mag'][0][1000:1400].T, S_hat[1000:1400].T, int(len(s)*0.27))

# Post-processing normalisation
s_hat = s_hat/1.1/np.max(np.abs(s_hat))
s_hat = s_hat[:len(z)]
fs = 16000
sf.write('clean.wav', s, fs)
sf.write('enhanced.wav', s_hat, fs)
sf.write('noisy.wav', z, fs)
# enhance_dir(data_dir, 'near', 4, net, device)
