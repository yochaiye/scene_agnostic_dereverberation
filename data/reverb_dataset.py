#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:48:25 2020

@author: yochai_yemini
"""

import numpy as np
from torch.utils.data import Dataset
from data.utils import normalize_log_spec
import pickle
import pathlib

eps = 2.2204*np.exp(-16)
K = 512
FRAMES_NUM = 256

class ReverbDataset(Dataset):
    '''
    Generates training samples and labels.
    The training samples are the multichannel reverberated speech ion the STFT
    domain, and the labels are the clean speech signals in the STFT domain.
    '''
    def __init__(self, mics_num, dataset, data_source, downsampling_factor):
        super().__init__()
        self.mics_num = mics_num
        self.source_to_path = {'train': 'train/random', 'val_near': 'val/near', 'val_far': 'val/far'}
        base_path = pathlib.Path(f'./spectrograms/{dataset}/mics{mics_num}')
        dataset_path = base_path / f'{self.source_to_path[data_source]}'
        min_max_file = base_path / 'train/global_min_max.p'
        self.files = [e for e in dataset_path.iterdir() if e.is_file()]
        self.files = self.files[::downsampling_factor]
        print(f'number of {self.source_to_path[data_source]} samples: {len(self.files)}')
        
        with open(min_max_file, 'rb') as f:
            self.log_max_clean, self.log_min_clean, self.log_max_reverb, self.log_min_reverb = pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        with open(file, 'rb') as f:
            reverb, clean = pickle.load(f)

        # Extract only the desired channels
        # reverb['mag'] = reverb['mag'][:self.mics_num]
        # reverb['phase'] = reverb['phase'][:self.mics_num]

        # |STFT| --> log(|STFT|)
        clean = np.log(clean['mag'] + eps)
        reverb = np.log(reverb['mag'] + eps)
        reverb = reverb[:self.mics_num]

        # Normalise spectrograms to be in [-1, 1]
        clean = normalize_log_spec(clean, self.log_max_clean, self.log_min_clean)
        reverb = normalize_log_spec(reverb, self.log_max_reverb, self.log_min_reverb)

        # Add a channel dimension for multi-channel sets
        if self.mics_num != 1:
            reverb = np.expand_dims(reverb, 1)
        clean = np.expand_dims(clean, 0)

        return reverb, clean
