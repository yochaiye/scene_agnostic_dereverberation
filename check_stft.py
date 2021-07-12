import numpy as np
from collections import OrderedDict
import scipy.io
import soundfile as sf
import pathlib

eps=2.2204*np.exp(-16)
MAT = scipy.io.loadmat('synt_win.mat')
synt_win = MAT['synt_win']
K = 512
overlap = 0.75

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
    
    
def log_abs_stft(z, K, overlap, eps):
    sub_num = 1 / (1 - overlap) - 1
    SEG_NO = np.fix(len(z) / (K * (1 - overlap))) - sub_num
    Z = np.zeros((int(K / 2 + 1), int(SEG_NO)))
    for seg in np.arange(1, SEG_NO + 1):
        time_cal = np.arange((seg - 1) * K * (1 - overlap) + 1,
                             (seg - 1) * K * (1 - overlap) + K + 1) - 1
        time_cal = time_cal.astype('int')
        V = np.fft.fft(z[time_cal] * np.append(np.hanning(K - 1), 0))
        time_freq = np.arange(1, K / 2 + 1 + 1) - 1
        time_freq = time_freq.astype('int')
        Z[:, int(seg - 1)] = np.log(np.abs(V[time_freq]) + eps)
    return Z

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

def enhancement_wo_mask(Rho_k,z,SEG_NO):

    s_est=np.zeros((len(z),1))

    P_full = np.zeros((K, SEG_NO))
    A_full = np.zeros((K, SEG_NO))
    Rho=Rho_k.T
    for seg in np.arange(1,SEG_NO+1):

        time_cal=np.arange((seg-1)*K*(1-overlap)+1,(seg-1)*K*(1-overlap)+K+1)-1
        time_cal=time_cal.astype('int64')
        temp=z[time_cal]*(np.append(np.hanning(K-1),0))
        Z11=np.fft.fft(temp)
        time_freq=np.arange(1,K/2+1+1)-1
        time_freq=time_freq.astype('int64')
        A1=np.log(np.abs(Z11[time_freq])+eps).reshape(257,1)
        P1=np.angle(Z11).reshape(512,1)
        P_full[:, seg - 1] = np.squeeze(P1)
        Ahat=np.exp(Rho)[:,seg-1].reshape(257,1)
        Ahat[0:3]=Ahat[0:3]*0.001
        inv_Ahat=Ahat[::-1]
        inv_Ahat=inv_Ahat[1:256]
        Ahatfull=np.append(Ahat,inv_Ahat).reshape(K,1)

        Z2=Ahatfull*np.exp(1j*P1)
        r1=np.real(np.fft.ifft(Z2.T))

        r2=r1.T*synt_win
        s_est[time_cal,:]=s_est[time_cal,:]+r2

    s_est=s_est/1.1/np.max(np.abs(s_est))
    return (s_est)


clean_path = '/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/c0m/c0mc0210.wav'
# reverb_path = '/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data/far_test/c3h/c3hc020p_ch1.wav'
s, fs = sf.read(clean_path)
s = s/1.1/np.max(np.abs(s))
S, P = stft(s, K, overlap, 'log')
s_hat = istft(S, P, synt_win)
s_hat = s_hat/1.1/np.max(np.abs(s_hat))


S_ori = log_abs_stft(s, K, overlap, eps)
s_hat_ori = enhancement_wo_mask(S_ori.T, s, S_ori.shape[1])
s_hat_ori = s_hat_ori.squeeze()
s_hat = s_hat[:len(s_hat_ori)]