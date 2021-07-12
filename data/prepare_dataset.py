#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:42:48 2020

@author: yochai_yemini
"""

import numpy as np
import soundfile as sf
from utils import stft, normalize_mc, update_max, update_min
import pathlib
import pickle
import argparse

# number of microphones used for recording
MICS_NUM = 8

# Leave this number of STFT frames from each recording
FRAMES_NUM = 256

# STFT parameters
K = 512
overlap = 0.75
eps = 2.2204*np.exp(-16)

#male_dir = ["c0b","c0e","c0k","c0m","c0o","c0p","c0q","c0t","c0w","c0x","c0y",
# "c1b","c1c","c1e","c1g","c1h","c1i","c1j","c1k","c1l","c1m","c1n","c1o","c1p",
# "c1r","c1s","c1t","c1u","c1v","c1w","c1x","c1y","c1z","c02","c2b","c2d","c2e",
# "c04","c05","c06","c08","c11","c14","c15","c16","c17","c20","c22","c23","c24",
# "c25","c27","c28"]
#female_dir = ["c0a","c0c","c0d","c0f","c0g","c0h","c0i","c0j","c0l","c0n",
# "c0r","c0s","c0u","c0v","c0z","c1a","c1d","c1f","c1q","c2a","c2c","c2f","c2g",
# "c2h","c2i","c2j","c2k","c2l","c03","c07","c09","c10","c12","c13","c18","c19",
# "c21","c26","c29"]


def save_spectrograms(mics_num, clean_dir, reverb_dir, save_dir, split):
    save_dir.mkdir(parents=True, exist_ok=True)

    speakers_dir_clean = sorted([e for e in clean_dir.iterdir() if e.is_dir()])
    # [print(speaker.name) for speaker in speakers_dir_clean]
    log_max_clean = 0
    log_max_reverb = 0
    log_min_clean = 1
    log_min_reverb = 1

    total_count = 0
    
    for speaker in speakers_dir_clean:
        
        # Get all clean WAV files from the speaker's dir
        speaker_files_clean = sorted([e for e in speaker.iterdir() if e.is_file() and e.suffix == '.wav'])
        
        # Get all reverb WAV files from the speaker's dir
        speaker_dir_reverb = reverb_dir / speaker.name
        speaker_files_reverb = sorted([e for e in speaker_dir_reverb.iterdir() if e.is_file() and e.suffix == '.wav'])
        
        assert len(speaker_files_clean) * MICS_NUM == len(speaker_files_reverb)
        
        # Iterate on WAV files
        for i in range(len(speaker_files_clean)):
            # Get clean wav
            file_clean = speaker_files_clean[i]
            print('processing %s / %s (%d)' % (speaker.name, file_clean.name, total_count+1))
            # print(str(file_clean))
            s, fs = sf.read(file_clean)
            s = s/1.1/np.max(np.abs(s))
            S_temp = {}
            S_temp['mag'], S_temp['phase'] = stft(s, K, overlap)
            
            if len(S_temp['mag'].T) < 256:
                continue
            
            # Get multichannel reverb wavs
            z = []
            for j in range(mics_num):
                file_reverb = speaker_files_reverb[j + i*MICS_NUM]
                # print(str(file_reverb))
                assert file_reverb.name.startswith(file_clean.name[:-4], 0)
                temp, fs = sf.read(file_reverb)
                z.append(temp)
                
            z = normalize_mc(np.array(z))
            # z = np.array(z)
            # z = z/1.1/np.max(np.abs(z))
            
            # Get the STFTs of the mutlichannel reverb signals
            Z_temp = [stft(z[m], K, overlap) for m in range(mics_num)]
            
            if split == 'train':
                if len(S_temp['mag'].T) < FRAMES_NUM*2:
                    rounds_num = 1
                else:
                    rounds_num = 2
            else:
                rounds_num = 1
            
            for _ in range(rounds_num):
                # fetch  256-frames slices from the spectrograms
                S = {}
                start_index = np.random.randint(S_temp['mag'].shape[1]-255)
                S['mag'] = S_temp['mag'][:-1, start_index:start_index+256].T
                # S['phase'] = S_temp['phase'][:-1, start_index:start_index+256].T
                S['mag'] = S['mag'].astype('float32')
                # S['phase'] = S['phase'].astype('float32')
                
                Z = {'mag': np.zeros((mics_num, FRAMES_NUM, int(K/2)))
                     }
                      # 'phase': np.zeros((mics_num, FRAMES_NUM, int(K/2)))}
                for j in range(mics_num):
                    Z['mag'][j] = Z_temp[j][0][:-1, start_index:start_index+FRAMES_NUM].T
                    # Z['phase'][j] = Z_temp[j][1][:-1, start_index:start_index+FRAMES_NUM].T
                Z['mag'] = Z['mag'].astype('float32')
                # Z['phase'] = Z['phase'].astype('float32')
                
                # Save spectrograms
                spectrogram_file = save_dir / '{}.p'.format(total_count)
                with open(spectrogram_file, 'wb') as f:
                    pickle.dump((Z, S), f)
                
                # Update global min and max values for the clean signal
                S_log_mag = np.log(S['mag']+eps)
                log_max_clean = update_max(log_max_clean, S_log_mag)
                log_min_clean = update_min(log_min_clean, S_log_mag)
                # print(np.log(np.max(S['mag'])+eps), np.log(np.min(S['mag']+eps)))
                print(log_max_clean, log_min_clean)
                
                # Update global min and max values for the reverb signals
                Z_log_mag = np.log(Z['mag']+eps)
                log_max_reverb = update_max(log_max_reverb, Z_log_mag)
                log_min_reverb = update_min(log_min_reverb, Z_log_mag)
                # print(np.log(np.max(Z['mag'])+eps), np.log(np.min(Z['mag'])+eps))
                print(log_max_reverb, log_min_reverb)
                
                total_count += 1
    return log_max_clean, log_min_clean, log_max_reverb, log_min_reverb

            
########### Process training data ###########
parser = argparse.ArgumentParser(
        description="Extract spectrograms from reverberant and clean WAV files for training the network."
    )
parser.add_argument('--mics_num', help='number of microphones to process in the array.', type=int, default=2)
parser.add_argument("--dataset", choices=['BIUREV', 'BIUREV-N'], default='BIUREV', help="spectrogrmas for BIUREV/BIUREV-N/")
args = parser.parse_args()
print('Processing training data')        

# Where to take the clean and reverberant audio files from
clean_wavs_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train')
reverb_wavs_dir = pathlib.Path(f'/mnt/dsi_vol1/users/yochai_yemini/{args.dataset}/train/random/')

# Where to save spectrograms
save_base_dir = pathlib.Path(f'./spectrograms/{args.dataset}/mics{args.mics_num}')
save_dir = save_base_dir / 'train/random'

log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = \
    save_spectrograms(args.mics_num, clean_wavs_dir, reverb_wavs_dir, save_dir, 'train')

# Save global min and max values
min_max_file = save_base_dir / 'train/global_min_max.p'
with open(min_max_file, 'wb') as f:
    pickle.dump((log_max_clean, log_min_clean, log_max_reverb, log_min_reverb), f)
    
########### Process validation data ########### 

print('Processing validation data: far')
clean_wavs_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_dt/data/cln_test')
reverb_wavs_base_dir = pathlib.Path(f'/mnt/dsi_vol1/users/yochai_yemini/{args.dataset}/val')
reverb_wavs_dir = reverb_wavs_base_dir / 'far'

save_dir = save_base_dir / 'val' / 'far'
save_spectrograms(args.mics_num, clean_wavs_dir, reverb_wavs_dir, save_dir, 'val')

print('Processing validation data: near')
reverb_wavs_dir = reverb_wavs_base_dir / 'near'
save_dir = save_base_dir / 'val' / 'near'
save_spectrograms(args.mics_num, clean_wavs_dir, reverb_wavs_dir, save_dir, 'val')
