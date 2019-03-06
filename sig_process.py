import os,re
import collections
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import pyworld as pw
# from reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp, mgc_to_mfsc, mfsc_to_mgc
# from acoufe import pitch
import soundfile as sf
import config
import utils

import librosa




def get_hcqt(audio):

    cqt_list = []
    shapes = []
    for h in config.harmonics:
        
        cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin*float(h), bins_per_octave = config.bins_per_octave)
        cqt_list.append(cqt.T)

    cqt_list = utils.match_time(cqt_list)
    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt

def f0_to_hertz(f0):
    # if f0 == 0:
    #     return 0
    # else:
    f0 = f0-69
    f0 = f0/12
    f0 = 2**f0
    f0 = f0*440
    return f0

def feats_to_audio(in_feats,filename, fs=config.fs,  mode=config.comp_mode):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    # f0[:,0] = f0[:,0]-69
    # f0[:,0] = f0[:,0]/12
    # f0[:,0] = 2**f0[:,0]
    # f0[:,0] = f0[:,0]*440
    f0[:,0] = f0_to_hertz(f0[:,0])

    f0 = f0[:,0]*(1-f0[:,1])


    if mode == 'mfsc':
        harm = mfsc_to_mgc(harm)
        ap = mfsc_to_mgc(ap)


    harm = mgc_to_sp(harm, 1025, 0.45)
    ap = mgc_to_sp(ap, 1025, 0.45)

    harm = 10**(harm/10)
    ap = 10**(ap/20)

    y=pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),int(fs),config.hoptime*1000)
    # import pdb;pdb.set_trace()
    sf.write(filename+'.wav',y,44100)

def stft_to_feats(vocals, fs, mode=config.comp_mode):
    feats=pw.wav2world(vocals,fs,frame_period=config.hoptime*1000)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10*np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    f0 = pitch.extract_f0_sac(vocals, fs, config.hoptime)

    y=69+12*np.log2(f0/440)
    # import pdb;pdb.set_trace()
    # y = hertz_to_new_base(f0)
    nans, x= utils.nan_helper(y)

    naners=np.isinf(y)

    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)

    if mode == 'mfsc':
        harmy=sp_to_mfsc(harm,60,0.45)
        apy=sp_to_mfsc(ap,4,0.45)
    elif mode == 'mgc':
        harmy=sp_to_mgc(harm,60,0.45)
        apy=sp_to_mgc(ap,4,0.45)
    out_feats=np.concatenate((harmy,apy,y.reshape((-1,2))),axis=1) 

    return out_feats

def get_feats(audio):



    stft = librosa.core.stft(audio, n_fft = config.nfft, hop_length = config.hopsize, window = config.window).T

    # voc_stft_mag = 2 * abs(voc_stft)/np.sum(config.window)

    # voc_stft_phase = np.angle(voc_stft)

    cqt = librosa.core.cqt(audio, sr = config.fs, hop_length = config.hopsize, n_bins = config.cqt_bins, fmin = config.fmin, bins_per_octave = config.bins_per_octave).T

    hcqt = get_hcqt(audio)

    hcqt = np.swapaxes(hcqt, 0,1)

    return stft, cqt, hcqt