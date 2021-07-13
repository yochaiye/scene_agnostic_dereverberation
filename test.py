#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:04:17 2020

@author: yochai_yemini
"""

import torch
import data.utils as utils
import networks.model_dss
import networks.model
import pathlib
import pickle
import numpy as np
import scipy.io
import soundfile as sf
import argparse
import matplotlib.pyplot as plt

MICS_NUM = 8
FRAMES_NUM = 256
K = 512
overlap = 0.75
eps = 2.2204 * np.exp(-16)

# Synthesis window for the ISTFT
MAT = scipy.io.loadmat('synt_win.mat')
synt_win = MAT['synt_win']


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
    magnitude = utils.normalize_log_spec(magnitude, log_max_reverb, log_min_reverb)

    magnitude = torch.from_numpy(magnitude)
    magnitude = magnitude.type(torch.FloatTensor)

    # Divide the input spectrogram into [256, 256] segments to match training
    edge_index = magnitude.shape[1] // FRAMES_NUM  # check how many 256-frames long segments fully fit
    residue = magnitude.shape[1] % FRAMES_NUM
    mics_num = magnitude.shape[0]
    # Concatenate segments along the batch dimension
    if mics_num == 1:
        to_model = magnitude[:, :edge_index * FRAMES_NUM, :-1].view(-1, 1, FRAMES_NUM, int(K / 2))
        if residue != 0:
            last_part = magnitude[:, -FRAMES_NUM:, :-1].view(-1, 1, FRAMES_NUM, int(K / 2))
            to_model = torch.cat([to_model, last_part], axis=0)
    else:
        if residue == 0:
            to_model = torch.zeros(edge_index, mics_num, 1, FRAMES_NUM, int(K / 2))
        else:
            to_model = torch.zeros(edge_index + 1, mics_num, 1, FRAMES_NUM, int(K / 2))
        trunc_mag = magnitude[:, :edge_index * FRAMES_NUM, :-1].unsqueeze(1)
        for i in range(edge_index):
            to_model[i] = trunc_mag[:, :, i * FRAMES_NUM:(i + 1) * FRAMES_NUM, :]
        if residue != 0:
            to_model[-1] = magnitude[:, -FRAMES_NUM:, :-1].unsqueeze(1)

    net.eval()
    with torch.no_grad():
        pred = net(to_model.to(device))
    pred = pred.squeeze().cpu()

    # assemble the enhanced magnitude
    if residue == 0:
        recon = pred.view(edge_index * FRAMES_NUM, int(K / 2))
    else:
        recon1 = pred[:-1, :].view(edge_index * FRAMES_NUM, int(K / 2))
        recon = torch.cat((recon1, pred[-1, -residue:, :]), axis=0)

    recon = torch.cat((recon, magnitude[0, :, -1:]), axis=1)  # Add the highest frequency

    # Denormalise
    recon = utils.denormalize_log_spec(recon, log_max_clean, log_min_clean)
    return recon.numpy()


def enhance_file(reverb_path, mics_num, net, device):
    '''
    Enhances a reverberant signal given its path

    Parameters
    ----------
    reverb_path : path to the reverbernat signal
    mics_num : number of microphones to use
    net : The trained enhancing network
    device : Device to run on (cpu or gpu)

    '''

    # Get multichannel reverb wavs
    z = []
    for j in range(mics_num):
        temp, fs = sf.read(str(reverb_path)[:-5] + '{}.wav'.format(j+1))
        z.append(temp)

    z = utils.normalize_mc(np.array(z))
    z = np.random.permutation(z)
    closest_mic = np.argmax(np.var(z, axis=-1))

    # Get the STFTs of the mutlichannel reverb signals
    temp = [utils.stft(z[m], K, overlap) for m in range(mics_num)]
    frames_num = len(temp[0][0].T)
    if frames_num < 256:
        raise RuntimeError('Recording is too short.')

    Z = {}
    Z['mag'] = np.zeros((mics_num, frames_num, int(K / 2 + 1)))
    Z['phase'] = np.zeros((mics_num, frames_num, int(K / 2 + 1)))
    for i in range(mics_num):
        Z['mag'][i] = temp[i][0].T
        Z['phase'][i] = temp[i][1].T

    # Enhance the log magnitude
    Z['mag'] = np.log(Z['mag'] + eps)
    S_hat = enhance_mag(Z['mag'], net, device)

    # Get the phase
    z = z[closest_mic]  # take only the first microphone for the phase
    z = z / 1.1 / np.max(np.abs(z))
    _, recon_phase = utils.stft(z, K, overlap)
    s_hat = utils.istft(S_hat.T, recon_phase, synt_win)


    # Post-processing normalisation
    s_hat = s_hat / 1.1 / np.max(np.abs(s_hat))
    s_hat = s_hat[:len(z)]
    return z, s_hat, Z['mag'][0].T.squeeze(), S_hat


def enhance_scenario(wavs_dir, results_dir, scenario_file, mics_num, net, device, trained_model_path):
    """
    Creates spectrograms of size 256x257 from the validation/test data

    Parameters
    ----------
    wavs_dir: directory with multichannel reverberant speech.
    results_dir : directory to save the enhanced files.
    scenario_file: text files listing all the reverberant files.
    mics_num: number of used microphones.
    net : trained pytorch model for dereverberation.
    device: cuda/cpu.
    trained_model_path: the path to the trained model file.
    """
    fs = 16000

    # Extract the reverbernat WAV files names
    with open(scenario_file, 'r') as f:
        reverb_files = f.readlines()
    reverb_files = [x.strip() for x in reverb_files]

    save_dir = results_dir / trained_model_path[17:42] # / scenario_file.name[:7]
    save_dir.mkdir(parents=True, exist_ok=True)

    total_count = 1

    for i, file in enumerate(reverb_files):
        reverb_path = wavs_dir / file[1:]
        print('processing %s (%d/%d)' %
              (reverb_path.name, total_count, len(reverb_files)))

        try:
            z, s_hat, _, _ = enhance_file(reverb_path, mics_num, net, device)
        except:     # In case the signal is too short
            continue
        enhanced_save_name = save_dir / file[1:]
        enhanced_save_name.parent.mkdir(parents=True, exist_ok=True)
        sf.write(enhanced_save_name, s_hat, fs)

        # L = np.min((len(s), len(z[0]), len(s_hat)))
        # save_spectrograms(S, Z['mag'], S_hat, L, save_dir)

        total_count += 1


if __name__ == '__main__':

    # Args parser
    parser = argparse.ArgumentParser('')

    parser.add_argument('--version_name', help='BIUREV/BIUREV-N', type=str)
    parser.add_argument('--dataset', help='BIUREV/BIUREV-N', type=str, default='BIUREV', choices=['BIUREV', 'BIUREV-N'])
    parser.add_argument('--ngf', help='base number of filters.', type=int, default=64)
    parser.add_argument('--unet_arch', help='U-net architecture.', type=str, default='vanilla',
                        choices=['dss', 'vanilla'])

    # gpu choice
    parser.add_argument('--gpu_id', help='gpu id to run on', type=int, default='0')

    args = parser.parse_args()

    # Load model
    trained_model_path = f"./trained_models/{args.version_name}/checkpoints/epoch69.pt"
    mics_num = int(trained_model_path[21])
    wavs_dir = pathlib.Path(f'/mnt/dsi_vol1/users/yochai_yemini/{args.dataset}')
    min_max_file = f'./spectrograms/BIUREV/mics{mics_num}/train/global_min_max.p'
    results_dir = pathlib.Path(f'/mnt/dsi_vol1/users/yochai_yemini/{args.dataset}/results')
    gpu = 3

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if mics_num != 1:
        if args.unet_arch == 'vanilla':
            net = networks.model.UNet(ngf=args.ngf, nc=mics_num, kernel_size=4).to(device)
        else:
            net = networks.model_dss.UNet(ngf=args.ngf, nc=1, kernel_size=4).to(device)
    else:
        net = networks.model.UNet(ngf=args.ngf, nc=1, kernel_size=4).to(device)
    net_state_dict = torch.load(trained_model_path, map_location=f"cuda:{args.gpu_id}")
    net_state_dict = utils.remove_module_str(net_state_dict)
    net.load_state_dict(net_state_dict)

    # Get the global min and max values for the spectrograms
    with open(min_max_file, 'rb') as f:
        log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = pickle.load(f)

    dists = ['near', 'far', 'random', 'winning_ticket']
    for dist in dists:
        scenario_file = pathlib.Path('./taskfiles/SimData_et_for_' + dist)
        scenario_file_clean = pathlib.Path('taskfiles/SimData_et_for_cln')
        enhance_scenario(wavs_dir, results_dir, scenario_file, mics_num, net, device, trained_model_path)

