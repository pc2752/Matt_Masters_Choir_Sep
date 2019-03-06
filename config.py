import numpy as np
import os
# import tensorflow as tf


wav_dir = '../datasets/ChoralSingingDataset/'


feats_dir = './feats/'
voc_feats_dir = './voc_feats/'


#FFT Parameters
fs = 44100.0
nfft = 1024
hopsize = 512
hoptime = hopsize/fs
window = np.hanning(nfft)


#CQT Parameters
fmin = 32.70
bins_per_octave = 60
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves
harmonics = [0.5, 1, 2, 3, 4, 5]

comp_mode = 'mfsc'

# Model parameters
batch_size = 30
max_phr_len = 30
max_models_to_keep = 10
log_dir = './log_all_but_1/'
init_lr = 0.0001
batches_per_epoch_train = 100
batches_per_epoch_val = 10
samples_per_file = 30
num_epochs = 3000
print_every = 1
validate_every = 1
save_every = 50
train_list = [x for x in os.listdir(feats_dir) if x.endswith('.hdf5') and not x.__contains__('1') and not x.startswith('nino')]
val_list = [x for x in os.listdir(feats_dir) if x.endswith('.hdf5') and x.__contains__('1') or x.startswith('nino')]
add_noise = False
noise_threshold = 0.2
wavenet_filters = 128
wavenet_layers = 5

# Sep Model
log_dir_sep = './log_sep/'