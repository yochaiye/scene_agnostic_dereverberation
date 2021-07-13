# Scene-Agnostic Dereverberation
PyTorch implementation of the 2021 INTERSPEECH paper "Scene-Agnostic Multi-Microphone Speech Dereverberation".  
The code for generating the BIUREV/BIUREV-N datasets described in the paper can be found [here](https://github.com/yochaiye/BIUREVgen).

## Prerequisites
- Acquire the BIUREV/BIUREV-N datasets.
- Python 3.
- [barbar](https://pypi.org/project/barbar/) package (pip install barbar).
- wandb (pip install wandb).

## Documentation
- `data` - a directory that handles data preparation and the PyTorch Dataset object.
- `networks` - a directory where the DSS and [Ernst et al.](https://arxiv.org/pdf/1803.08243.pdf) models are defined.
- `train.py` - code for training a network.
- `test.py` - code for evaluating a network.
- `losses.py` - implements the loss functions.
- `taskfiles` - text files comprising lists of WAV files used for evaluation (in essence, all test files in BIUREV/BIUREV-N datasets).

## Usage
#### 1. Dataset Preparation
* Extract spectrograms from the BIUREV/BIUREV-N datasets to be used for training and validation.  
  In `data/prepare_dataset.py`, set the variables `clean_wavs_dir` and `reverb_wavs_dir` as the directories where the clean and reverberant speech signals are stored.
  Makes sure to change the variables for both training(lines 154-155) and evaluation (lines 168-169) parts.
* Run:  
  ```
  python data/prepare_dataset.py --mics_num <desired number of microphones> --dataset <BIUREV or BIUREV-N>
  ```
  
  This can take from a few minutes to a couple of hours, depending on the number of microphones.
  
 #### 2. Training
 * Run:  
```
python train.py --mics_num <choose from 1-8> --dataset <BIUREV or BIUREV-N> --unet_arch <dss or vanilla> --gpu_ids <for example 0 or 0,1,2,3>
```

This will create a new directory, named `trained_networks`.  

#### 3. Evaluation:
* Change line 208 such that the variable `wavs_dir` refers to the parent directory of BIUREV/BIUREV-N.
* Run:
```
python test.py --version_name <name of version> --dataset <BIUREV or BIUREV-N> -unet_arch <dss or vanilla>.
```
For example:
```
python test.py --version_name mics8_24.03.2021_05:33:29 --dataset BIUREV -unet_arch dss.
```
The number of microphones is inferred from the version's name.

