# from __future__ import division
import numpy as np
import librosa
import os
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

import h5py

import config

import sig_process
import utils
from acoufe import pitch

from scipy.ndimage import filters
import itertools
# import essentia_backend as es

def process_f0(f0, f_bins, n_freqs):
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))

    haha = np.digitize(f0, f_bins) - 1

    idx2 = haha < n_freqs

    haha = haha[idx2]

    freqz[range(len(haha)), haha] = 1

    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T

    min_target = np.min(atb[range(len(haha)), haha])

    atb = atb / min_target

    # import pdb;pdb.set_trace()

    atb[atb > 1] = 1

    return atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins

def prep_DeepConvSep():
    x = [1,2,3,4]

    combos = [p for p in itertools.product(x, repeat=4)]

    import pdb;pdb.set_trace()

    songs = next(os.walk(config.wav_dir))[1]


    for song in songs:
        print ("Processing song %s" % song)
        song_dir = config.wav_dir+song+'/IndividualVoices/'
        singers = [x for x in os.listdir(song_dir) if x.endswith('.wav') and not x.startswith('.')]

        song_name = [x.split('_')[0] for x in singers][0]

        count = 0

        for combo in combos:
            combo_str = str(combo[0]) + str(combo[1]) + str(combo[2]) + str(combo[3])

            if not os.path.isfile(config.feats_dir+song_name+'_'+combo_str+'.hdf5'):
                if combo[0]!=0:
                    audio_sop, fs = librosa.core.load(os.path.join(song_dir,song_name+'_soprano_'+str(combo[0])+'.wav'), sr = config.fs)
                    stft_sop = librosa.core.stft(audio_sop, n_fft=config.nfft, hop_length=config.hopsize, window=config.window).T
                if combo[1]!=0:
                    audio_alt, fs = librosa.core.load(os.path.join(song_dir,song_name + '_alto_' + str(combo[1]) + '.wav'), sr=config.fs)
                    stft_alt = librosa.core.stft(audio_alt, n_fft=config.nfft, hop_length=config.hopsize,
                                                 window=config.window).T
                if combo[2]!=0:
                    audio_bas, fs = librosa.core.load(os.path.join(song_dir, song_name + '_bass_' + str(combo[2]) + '.wav'), sr=config.fs)
                    stft_bas = librosa.core.stft(audio_bas, n_fft=config.nfft, hop_length=config.hopsize,
                                                 window=config.window).T
                if combo[3]!=0:
                    audio_ten, fs = librosa.core.load(os.path.join(song_dir, song_name + '_tenor_' + str(combo[3]) + '.wav'), sr=config.fs)
                    stft_ten = librosa.core.stft(audio_ten, n_fft=config.nfft, hop_length=config.hopsize,
                                                 window=config.window).T

                part_stft = [stft_sop, stft_alt, stft_bas, stft_ten]
                part_stft = np.stack(utils.match_time(part_stft))

                part_stft = np.moveaxis(part_stft, 0, -1)

                audio = np.stack(utils.match_time([audio_sop,audio_alt, audio_bas, audio_ten])).sum(axis=0)/4

                voc_stft = librosa.core.stft(audio, n_fft=config.nfft, hop_length=config.hopsize,
                                                 window=config.window).T

                assert voc_stft.shape[0] == part_stft.shape[0]


                hdf5_file = h5py.File(config.feats_dir+song_name+'_'+combo_str+'.hdf5', mode='w')

                hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.complex64)

                hdf5_file.create_dataset("part_stft", part_stft.shape, np.complex64)




                hdf5_file["voc_stft"][:,:] = voc_stft

                hdf5_file["part_stft"][:,:] = part_stft



                hdf5_file.close()


            count+=1

            utils.progress(count,len(combos))




if __name__ == '__main__':
    prep_DeepConvSep()