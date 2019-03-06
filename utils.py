import numpy as np
import sys
import config
import scipy
import csv
import librosa
# import mir_eval
import pandas as pd

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def match_time(feat_list):
    """ 
    Matches the shape across the time dimension of a list of arrays.
    Assumes that the first dimension is in time, preserves the other dimensions
    """
    shapes = [f.shape for f in feat_list]
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[0] for s in shapes])
        new_list = []
        for i in range(len(feat_list)):
            new_list.append(feat_list[i][:min_time])
        feat_list = new_list
    return feat_list

def csv_to_list(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        return your_list

def read_f0_file(file_name):
    ref_time, ref_freqs = mir_eval.io.load_ragged_time_series(file_name)
    return ref_time, ref_freqs  

def remove_zeros(ref_times_ori, ref_freqs_ori):
    for i, (tms, fqs) in enumerate(zip(ref_times_ori, ref_freqs_ori)):
        if any(fqs == 0):
            ref_freqs_ori[i] = np.array([f for f in fqs if f > 0])

    return ref_freqs_ori

def save_scores_mir_eval(scores, file_name_csv):
    scores = pd.DataFrame.from_dict(scores)
    scores.to_csv(file_name_csv)

def progress(count, total, suffix=''):
    """
    Helper function to print a progress bar.


    """
    
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def generate_overlapadd(allmix, time_context=config.max_phr_len, overlap=config.max_phr_len / 2,
                        batch_size=config.batch_size):
    # window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    input_size = allmix.shape[-1]

    i = 0
    start = 0
    while (start + time_context) < allmix.shape[0]:
        i = i + 1
        start = start - overlap + time_context
    fbatch = np.zeros([int(np.ceil(float(i) / batch_size)), batch_size, time_context, input_size]) + 1e-10

    i = 0
    start = 0

    while (start + time_context) < allmix.shape[0]:
        fbatch[int(i / batch_size), int(i % batch_size), :, :] = allmix[int(start):int(start + time_context), :]
        i = i + 1  # index for each block
        start = start - overlap + time_context  # starting point for each block

    return fbatch, i


def overlapadd(fbatch, nchunks, overlap=int(config.max_phr_len / 2)):
    input_size = fbatch.shape[-1]
    time_context = fbatch.shape[-2]
    batch_size = fbatch.shape[1]

    # window = np.sin((np.pi*(np.arange(2*overlap+1)))/(2.0*overlap))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window, window[::-1]))
    # time_context = net.network.find('hid2', 'hh').size
    # input_size = net.layers[0].size  #input_size is the number of spectral bins in the fft
    window = np.repeat(np.expand_dims(window, axis=1), input_size, axis=1)

    sep = np.zeros((int(nchunks * (time_context - overlap) + time_context), input_size))

    i = 0
    start = 0
    while i < nchunks:
        # import pdb;pdb.set_trace()
        s = fbatch[int(i / batch_size), int(i % batch_size), :, :]

        # print s1.shape
        if start == 0:
            sep[0:time_context] = s

        else:
            # print start+overlap
            # print start+time_context
            sep[int(start + overlap):int(start + time_context)] = s[overlap:time_context]
            sep[start:int(start + overlap)] = window[overlap:] * sep[start:int(start + overlap)] + window[:overlap] * s[
                                                                                                                      :overlap]
        i = i + 1  # index for each block
        start = int(start - overlap + time_context)  # starting point for each block
    return sep

def get_multif0(pitch_activation_mat, freq_grid, time_grid, thresh=0.3):
    """
    This function was taken from https://github.com/rabitt/ismir2017-deepsalience
    Compute multif0 output containing all peaks in the output that
       fall above thresh
    Parameters
    ----------
    pitch_activation_mat : np.ndarray
        Deep salience prediction
    freq_grid : np.ndarray
        Frequency values
    time_grid : np.ndarray
        Time values
    thresh : float, default=0.3
        Likelihood threshold
    Returns
    -------
    times : np.ndarray
        Time values
    freqs : list
        List of lists of frequency values
    """
    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(time_grid))]
    # import pdb;pdb.set_trace()
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freq_grid[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return time_grid, est_freqs

def save_multif0_output(times, freqs, output_path):
    """
    This function was taken from https://github.com/rabitt/ismir2017-deepsalience
    save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)

def process_output(atb):
    freq_grid = librosa.cqt_frequencies(config.cqt_bins, config.fmin, config.bins_per_octave)
    time_grid = np.linspace(0, config.hoptime * atb.shape[0], atb.shape[0])
    time_grid, est_freqs = get_multif0(atb.T, freq_grid, time_grid)
    return time_grid, est_freqs