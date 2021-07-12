#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:39:18 2020

@author: yochai_yemini
"""

import torch
import numpy as np
from collections import OrderedDict

eps=2.2204*np.exp(-16)

def stft(z, K, overlap, spec_mode='abs'):
    if spec_mode not in ['abs', 'log']:
        raise ValueError("spec_mode must be one of ")
    sub_num = 1 / (1 - overlap) - 1
    SEG_NO = np.fix(len(z) / (K * (1 - overlap))) - sub_num
    Z = np.zeros((int(K / 2 + 1), int(SEG_NO)))
    P = np.zeros((int(K / 2 + 1),int(SEG_NO)))
    for seg in np.arange(1, SEG_NO + 1):
        time_cal = np.arange((seg - 1) * K * (1 - overlap) + 1,
                             (seg - 1) * K * (1 - overlap) + K + 1) - 1
        time_cal = time_cal.astype('int')
        V = np.fft.fft(z[time_cal] * np.append(np.hanning(K - 1), 0))
        time_freq = np.arange(1, K / 2 + 1 + 1) - 1
        time_freq = time_freq.astype('int')
        P[:,int(seg-1)] = np.angle(V[time_freq])
        Z[:, int(seg - 1)] = np.abs(V[time_freq])

    if spec_mode == 'abs':
        return Z, P
    elif spec_mode == 'log':
        return np.log(Z + eps), P


def istft(A, P, synt_win):
    '''
    Returns the ISTFT of a FREQ x TIME signal
    param: A is the log-magnitude of the STFT, dimensions are FREQ x TIME.
    param: P is the phase is the phase of the STFT, dimensions are FREQ X TIME.
    param: synt_win is the synthesis window of length FREQ.
    '''
    # s_est=np.zeros((len(z),1))
    K = 512
    overlap = 0.75
    
    #Switch back from log-magnitude to magnitude
    A = np.exp(A)
    A[0:3] *= 0.001
    
    # Create the a KxTIME STFT from the (K/2+1)xTIME STFT
    A_inv, P_inv = A[::-1], P[::-1]
    A_inv, P_inv = A_inv[1:-1], P_inv[1:-1]
    P_full = np.concatenate([P, -P_inv], 0)
    A_full = np.concatenate([A, A_inv], 0)
    # A_full, P_full = A_full.numpy(), P_full.numpy()
    
    # Attach the phase with the magnitude
    S = A_full * np.exp(1j*P_full)
    S = np.real(np.fft.ifft(S, axis=0))
    S = synt_win * S
    
    SEG_NO = S.shape[1]
    long = 0.008*16000*SEG_NO + K
    long = int(long)
    s_est = np.zeros((long,1))
    for seg in np.arange(1,SEG_NO+1):
        time_cal = np.arange((seg-1)*K*(1-overlap)+1,(seg-1)*K*(1-overlap)+K+1)-1
        time_cal = time_cal.astype('int64')
        s_est[time_cal,:] = s_est[time_cal,:] + np.expand_dims(S[:, seg-1], axis=1)

    return s_est.squeeze()


def normalize_mc(z, desired_rms = 0.1, eps = 1e-4):
    '''
    Parameters
    ----------
    samples : numpy array comprising the multichannel reverberated speech
    desired_rms : desired rms after notmalisation. The default is 0.1.
                  rms is calculated over all samples and across channels.
    eps : The default is 1e-4.

    Returns
    -------
    samples : the normalised multichannel signals.

    '''
    if len(z) == 1:
        z = z/1.1/np.max(np.abs(z))
    else:
        rms = np.maximum(eps, np.sqrt(np.mean(z**2)))
        z = z * (desired_rms / rms)
    return z


def update_max(max_val, S):
    poten_max = np.max(S)
    if poten_max > max_val:
        return poten_max
    else:
        return max_val


def update_min(min_val, S):
    poten_min = np.min(S)
    if poten_min < min_val:
        return poten_min
    else:
        return min_val


def normalize_log_spec(z, max_val, min_val):
    normalized_z = (2*(z-max_val))/(max_val-min_val) + 1
    return normalized_z


def denormalize_log_spec (normelized_z, max_num = 1.0, min_num = 0.0):
    z = (normelized_z-1)*(max_num-min_num)/2 + max_num
    return z


def remove_module_str(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename
