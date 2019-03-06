import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils
import librosa
from scipy.ndimage import filters


def process_file(file_name):

    feat_file = h5py.File(config.feats_dir + file_name, 'r')

    atb = feat_file['atb'][()]

    atb = atb[:,1:]

    hcqt = feat_file['voc_hcqt'][()]

    feat_file.close()

    # import pdb;pdb.set_trace()

    return atb, np.array(hcqt)

def data_gen(mode = 'Train', sec_mode = 0):

    if mode == 'Train' :
        num_batches = config.batches_per_epoch_train
        file_list = config.train_list
        # import pdb;pdb.set_trace()
    else:
        file_list = config.val_list
        num_batches = config.batches_per_epoch_val


    max_files_to_process = int(config.batch_size / config.samples_per_file)



    for k in range(num_batches):

        out_mix = []
        out_part = []


        for i in range(max_files_to_process):

            voc_index = np.random.randint(0, len(file_list))
            voc_file = file_list[voc_index]
            # atb, hcqt = process_file(voc_file)
            feat_file = h5py.File(config.feats_dir + voc_file, 'r')

            mix_stft = feat_file['voc_stft'][()]

            part_stft = feat_file['part_stft'][()]



            for j in range(config.samples_per_file):
                voc_idx = np.random.randint(0, len(mix_stft) - config.max_phr_len)
                out_mix.append(mix_stft[voc_idx:voc_idx + config.max_phr_len])
                out_part.append(part_stft[voc_idx:voc_idx + config.max_phr_len])

            feat_file.close()


        yield abs(np.array(out_mix)).reshape(config.batch_size, config.max_phr_len,-1,1), abs(np.array(out_part))






def get_stats():
    voc_list = [x for x in os.listdir(config.voc_feats_dir) if x.endswith('.hdf5')]


    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000
 

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voc_feats_dir+voc_to_open, "r")

        # import pdb;pdb.set_trace()

        feats = voc_file["voc_feats"][()]

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0



        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i]   

    import pdb;pdb.set_trace()


    hdf5_file = h5py.File('./stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   


    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat


    hdf5_file.close()





def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen('val', sec_mode = 0)
    while True :
        start_time = time.time()
        ins, outs = next(gen)
        print(time.time()-start_time)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()