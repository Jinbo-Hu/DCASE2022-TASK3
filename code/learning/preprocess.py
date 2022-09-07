import shutil
import sys
from functools import reduce
from pathlib import Path
from timeit import default_timer as timer

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.data import BaseDataset, collate_fn
from methods.feature import Features_Extractor_MIC
from methods.utils.data_utilities import (
    _segment_index, convert_output_format_polar_to_cartesian,
    load_output_format_file)
from utils.common import float_samples_to_int16, int16_samples_to_float32
from utils.config import get_afextractor


class Preprocess:
    
    """Preprocess the audio data.

    1. Extract wav file and store to hdf5 file
    2. Extract meta file and store to hdf5 file
    """
    
    def __init__(self, args, cfg, dataset):
        """
        Args:
            args: parsed args
            cfg: configurations
            dataset: dataset class
        """
        self.args = args
        self.cfg = cfg
        self.dataset = dataset
        self.fs = cfg['data']['sample_rate']
        self.n_fft = cfg['data']['nfft']
        self.n_mels = cfg['data']['n_mels']
        self.hoplen = cfg['data']['hoplen']
            
        # Path for dataset
        hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset'])

        # Path for extraction of wav
        self.data_dir_list = [
            dataset.dataset_dir[args.dataset_type]['foa'][args.dataset],
            dataset.dataset_dir[args.dataset_type]['mic'][args.dataset],
        ]
        data_h5_dir = hdf5_dir.joinpath('data').joinpath('{}fs'.format(self.fs))
        wav_h5_dir = data_h5_dir.joinpath('wav')
        self.wav_h5_dir_list = [
            wav_h5_dir.joinpath(args.dataset_type).joinpath('foa').joinpath(args.dataset),
            wav_h5_dir.joinpath(args.dataset_type).joinpath('mic').joinpath(args.dataset),
        ]
        self.data_statistics_path_list = [
            wav_h5_dir.joinpath(args.dataset_type).joinpath('foa').joinpath(args.dataset+'statistics_foa.txt'),
            wav_h5_dir.joinpath(args.dataset_type).joinpath('mic').joinpath(args.dataset+'statistics_mic.txt')
        ]

        # Path for extraction of label
        label_dir = hdf5_dir.joinpath('label')
        self.meta_dir_list = dataset.dataset_dir[args.dataset_type]['meta'][args.dataset]
        self.meta_pit_dir = label_dir.joinpath('track_pit_ov'+str(dataset.max_ov)+'of5_discontinuous')\
            .joinpath(args.dataset_type).joinpath(args.dataset)
        self.meta_sed_dir = label_dir.joinpath('sed').joinpath(args.dataset)
        self.meta_adpit_dir = label_dir.joinpath('adpit').joinpath(args.dataset_type).joinpath(args.dataset)

        # Path for extraction of features
        self.feature_h5_dir = data_h5_dir.joinpath('feature').joinpath(args.dataset_type).joinpath(cfg['data']['audio_feature'])

        # Path for indexes of data
        self.data_type = 'wav' if self.cfg['data']['audio_feature'] in ['logmelIV', 'logmel'] else 'feature'
        self.channels_dict = {'logmel': 4, 'logmelIV': 7}
        self.indexes_path_list = [ 
            data_h5_dir.joinpath(self.data_type).joinpath('{}set_{}sChunklen_{}sHoplen_train.csv'\
                .format(args.dataset_type, cfg['data']['train_chunklen_sec'], cfg['data']['train_hoplen_sec'])),
            data_h5_dir.joinpath(self.data_type).joinpath('{}set_{}sChunklen_{}sHoplen_test.csv'\
                .format(args.dataset_type, cfg['data']['test_chunklen_sec'], cfg['data']['test_hoplen_sec']))]
        
        # Path for scalar
        self.scalar_h5_dir = data_h5_dir.joinpath('scalar')
        dataset_name = '_'.join(sorted(str(cfg['dataset_synth']).split(',')))
        fn_scalar = '{}_nfft{}_hop{}_mel{}_{}.h5'.format(cfg['data']['audio_feature'], \
            self.n_fft, self.hoplen, self.n_mels, dataset_name)
        self.scalar_path = self.scalar_h5_dir.joinpath(fn_scalar)
            
        
    def extract_data(self):
        """ Extract wave and store to hdf5 file
        """

        print('Converting wav file to hdf5 file starts......\n')
        
        for h5_dir in self.wav_h5_dir_list:
            if h5_dir.is_dir():
                flag = input("HDF5 folder {} is already existed, delete it? (y/n)".format(h5_dir)).lower()
                if flag == 'y':
                    shutil.rmtree(h5_dir)
                elif flag == 'n':
                    print("User select not to remove the HDF5 folder {}. The process will quit.\n".format(h5_dir))
                    return
            h5_dir.mkdir(parents=True)
        for statistic_path in self.data_statistics_path_list:
            if statistic_path.is_file():
                statistic_path.unlink()

        for idx, data_dir in enumerate(self.data_dir_list):
            begin_time = timer()
            h5_dir = self.wav_h5_dir_list[idx]
            statistic_path = self.data_statistics_path_list[idx]
            audio_count = 0
            silent_audio_count = 0
            data_list = [path for data_subdir in data_dir for path in sorted(data_subdir.glob('**/*.wav')) if not path.name.startswith('.')]
            iterator = tqdm(data_list, total=len(data_list), unit='it')
            for data_path in iterator:
                # read data
                data, _ = librosa.load(data_path, sr=self.fs, mono=False)
                if len(data.shape) == 1:
                    data = data[None,:]
                '''data: (channels, samples)'''

                # silent data statistics
                lst = np.sum(np.abs(data), axis=1) > data.shape[1]*1e-4
                if not reduce(lambda x, y: x*y, lst):
                    with statistic_path.open(mode='a+') as f:
                        print(f"Silent file in feature extractor: {data_path.name}\n", file=f)
                        silent_audio_count += 1
                        tqdm.write("Silent file in feature extractor: {}".format(data_path.name))
                        tqdm.write("Total silent files are: {}\n".format(silent_audio_count))

                # save to h5py
                h5_path = h5_dir.joinpath(data_path.stem + '.h5')
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset(name='waveform', data=float_samples_to_int16(data), dtype=np.int16)

                audio_count += 1

                tqdm.write('{}, {}, {}'.format(audio_count, h5_path, data.shape))

            with statistic_path.open(mode='a+') as f:
                print(f"Total number of audio clips extracted: {audio_count}", file=f)
                print(f"Total number of silent audio clips is: {silent_audio_count}\n", file=f)

            iterator.close()
            print("Extacting feature finished! Time spent: {:.3f} s".format(timer() - begin_time))
    

    def extract_ADPIT_label(self):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)
        """

        def _get_adpit_labels_for_file(_desc_file):
            """
            Reads description file and returns classification based SED labels and regression based DOA labels
            for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

            :param _desc_file: dcase format of the meta file
            :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
            """

            _nb_label_frames = list(_desc_file.keys())[-1]
            _nb_lasses = self.dataset.num_classes
            se_label = np.zeros((_nb_label_frames, 6, _nb_lasses))  # [nb_frames, 6, max_classes]
            x_label = np.zeros((_nb_label_frames, 6, _nb_lasses))
            y_label = np.zeros((_nb_label_frames, 6, _nb_lasses))
            z_label = np.zeros((_nb_label_frames, 6, _nb_lasses))

            for frame_ind, active_event_list in _desc_file.items():
                if frame_ind < _nb_label_frames:
                    active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                    active_event_list_per_class = []
                    for i, active_event in enumerate(active_event_list):
                        active_event_list_per_class.append(active_event)
                        if i == len(active_event_list) - 1:  # if the last
                            if len(active_event_list_per_class) == 1:  # if no ov from the same class
                                # a0----
                                active_event_a0 = active_event_list_per_class[0]
                                se_label[frame_ind, 0, active_event_a0[0]] = 1
                                x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[1]
                                y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                                z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                                # --b0--
                                active_event_b0 = active_event_list_per_class[0]
                                se_label[frame_ind, 1, active_event_b0[0]] = 1
                                x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[1]
                                y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                                z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                                # --b1--
                                active_event_b1 = active_event_list_per_class[1]
                                se_label[frame_ind, 2, active_event_b1[0]] = 1
                                x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[1]
                                y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                                z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            else:  # if ov with more than 2 sources from the same class
                                # ----c0
                                active_event_c0 = active_event_list_per_class[0]
                                se_label[frame_ind, 3, active_event_c0[0]] = 1
                                x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[1]
                                y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                                z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                                # ----c1
                                active_event_c1 = active_event_list_per_class[1]
                                se_label[frame_ind, 4, active_event_c1[0]] = 1
                                x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[1]
                                y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                                z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                                # ----c2
                                active_event_c2 = active_event_list_per_class[2]
                                se_label[frame_ind, 5, active_event_c2[0]] = 1
                                x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[1]
                                y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                                z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]

                        elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                            if len(active_event_list_per_class) == 1:  # if no ov from the same class
                                # a0----
                                active_event_a0 = active_event_list_per_class[0]
                                se_label[frame_ind, 0, active_event_a0[0]] = 1
                                x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[1]
                                y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                                z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                                # --b0--
                                active_event_b0 = active_event_list_per_class[0]
                                se_label[frame_ind, 1, active_event_b0[0]] = 1
                                x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[1]
                                y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                                z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                                # --b1--
                                active_event_b1 = active_event_list_per_class[1]
                                se_label[frame_ind, 2, active_event_b1[0]] = 1
                                x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[1]
                                y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                                z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            else:  # if ov with more than 2 sources from the same class
                                # ----c0
                                active_event_c0 = active_event_list_per_class[0]
                                se_label[frame_ind, 3, active_event_c0[0]] = 1
                                x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[1]
                                y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                                z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                                # ----c1
                                active_event_c1 = active_event_list_per_class[1]
                                se_label[frame_ind, 4, active_event_c1[0]] = 1
                                x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[1]
                                y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                                z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                                # ----c2
                                active_event_c2 = active_event_list_per_class[2]
                                se_label[frame_ind, 5, active_event_c2[0]] = 1
                                x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[1]
                                y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                                z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            active_event_list_per_class = []
            label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]

            return label_mat
        
        meta_list = [path for subdir in self.meta_dir_list for path in sorted(subdir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='it')
        self.meta_adpit_dir.mkdir(parents=True, exist_ok=True)
        for idx, meta_file in iterator:
            fn = meta_file.stem
            meta_dcase_format = load_output_format_file(meta_file)
            meta_dcase_format = convert_output_format_polar_to_cartesian(meta_dcase_format)
            meta_adpit = _get_adpit_labels_for_file(meta_dcase_format)
            meta_h5_path = self.meta_adpit_dir.joinpath(fn + '.h5')
            with h5py.File(meta_h5_path, 'w') as hf:
                hf.create_dataset(name='adpit', data=meta_adpit, dtype=np.float32)
            tqdm.write('{}, {}'.format(idx, meta_h5_path))   

    
    def extract_PIT_label(self):
        """ Extract track label for permutation invariant training. Store to h5 file
        """
        num_tracks = 5
        num_classes = self.dataset.num_classes
        meta_list = [path for subdir in self.meta_dir_list for path in sorted(subdir.glob('*.csv')) if not path.name.startswith('.')]
        iterator = tqdm(enumerate(meta_list), total=len(meta_list), unit='it')
        self.meta_pit_dir.mkdir(parents=True, exist_ok=True)
        for idx, meta_file in iterator:
            fn = meta_file.stem
            df = pd.read_csv(meta_file, header=None, sep=',')
            df = df.values
            num_frames = df[-1, 0] + 1
            sed_label = np.zeros((num_frames, num_tracks, num_classes))
            doa_label = np.zeros((num_frames, num_tracks, 3))
            event_indexes = np.array([[None] * num_tracks] * num_frames)  # event indexes of all frames
            track_numbers = np.array([[None] * num_tracks] * num_frames)   # track number of all frames
            for row in df:
                frame_idx = row[0]
                event_idx = row[1]
                track_number = row[2]                
                azi = row[3]
                elev = row[4]
                
                ##### track indexing #####
                # default assign current_track_idx to the first available track
                current_event_indexes = event_indexes[frame_idx]
                current_track_indexes = np.where(current_event_indexes == None)[0].tolist()
                # if current_track_indexes:
                #     continue
                current_track_idx = current_track_indexes[0]    

                # tracking from the last frame if the last frame is not empty
                # last_event_indexes = np.array([None] * num_tracks) if frame_idx == 0 else event_indexes[frame_idx-1]
                # last_track_indexes = np.where(last_event_indexes != None)[0].tolist() # event index of the last frame
                # last_events_tracks = list(zip(event_indexes[frame_idx-1], track_numbers[frame_idx-1]))
                # if last_track_indexes != []:
                #     for track_idx in last_track_indexes:
                #         if last_events_tracks[track_idx] == (event_idx, track_number):
                #             if current_track_idx != track_idx:  # swap tracks if track_idx is not consistant
                #                 sed_label[frame_idx, [current_track_idx, track_idx]] = \
                #                     sed_label[frame_idx, [track_idx, current_track_idx]]
                #                 doa_label[frame_idx, [current_track_idx, track_idx]] = \
                #                     doa_label[frame_idx, [track_idx, current_track_idx]]
                #                 event_indexes[frame_idx, [current_track_idx, track_idx]] = \
                #                     event_indexes[frame_idx, [track_idx, current_track_idx]]
                #                 track_numbers[frame_idx, [current_track_idx, track_idx]] = \
                #                     track_numbers[frame_idx, [track_idx, current_track_idx]]
                #                 current_track_idx = track_idx
                #########################

                # label encode
                azi_rad, elev_rad = azi * np.pi / 180, elev * np.pi / 180
                sed_label[frame_idx, current_track_idx, event_idx] = 1.0
                doa_label[frame_idx, current_track_idx, :] = np.cos(elev_rad) * np.cos(azi_rad), \
                    np.cos(elev_rad) * np.sin(azi_rad), np.sin(elev_rad)
                event_indexes[frame_idx, current_track_idx] = event_idx
                track_numbers[frame_idx, current_track_idx] = track_number

            meta_h5_path = self.meta_pit_dir.joinpath(fn + '.h5')
            with h5py.File(meta_h5_path, 'w') as hf:
                hf.create_dataset(name='sed_label', data=sed_label[:, :self.dataset.max_ov, :], dtype=np.float32)
                hf.create_dataset(name='doa_label', data=doa_label[:, :self.dataset.max_ov, :], dtype=np.float32)
            
            tqdm.write('{}, {}'.format(idx, meta_h5_path))   


    def extract_mic_features(self):
        """ Extract features from MIC format signals
        """

        print('Extracting {} features starts......\n'.format(self.cfg['data']['audio_feature']))
        if self.feature_h5_dir.is_dir():
            flag = input("HDF5 folder {} is already existed, delete it? (y/n)".format(self.feature_h5_dir)).lower()
            if flag == 'y':
                shutil.rmtree(self.feature_h5_dir)
            elif flag == 'n':
                print("User select not to remove the HDF5 folder {}. The process will quit.\n".format(self.feature_h5_dir))
                return
        self.feature_h5_dir.mkdir(parents=True)
        af_extractor_mic = Features_Extractor_MIC(self.cfg)
        mic_path_list = sorted(self.wav_h5_dir_list[1].glob('*.h5'))
        iterator = tqdm(enumerate(mic_path_list), total=len(mic_path_list), unit='it')
        for count, file in iterator:
            fn = file.stem
            feature_path =self.feature_h5_dir.joinpath(fn+'.h5')
            with h5py.File(file, 'r') as hf:
                waveform = int16_samples_to_float32(hf['waveform'][:]).T
            nb_feat_frams = int(len(waveform) / self.hoplen)
            spect = af_extractor_mic._spectrogram(waveform, nb_feat_frams)
            # spect: [n_frames, n_freqs, n_chs]
            if self.cfg['data']['audio_feature'] == 'logmelgcc':
                logmel_spec = af_extractor_mic._get_logmel_spectrogram(spect)
                # logmel_spec: [n_frames, n_mels, n_chs]
                gcc = af_extractor_mic._get_gcc(spect)
                # gcc: [n_frames, n_mels, n_chs]
                feature = np.concatenate((logmel_spec, gcc), axis=-1).transpose((2,0,1))
                # feature: [n_chs, n_frames, n_mels]
                print('feature shape: ', feature.shape)
            elif self.cfg['data']['audio_feature'] == 'salsalite':
                feature = af_extractor_mic._get_salsalite(spect)
            with h5py.File(feature_path, 'w') as hf:
                hf.create_dataset('feature', data=feature, dtype=np.float32)
            tqdm.write('{}, {}, features: {}'.format(count, fn, feature.shape))
        iterator.close()
        print('Extracting {} features finished!'.format(self.cfg['data']['audio_feature']))


    def extract_index(self):
        """Extract index of clips for training and testing
        """

        chunklen_sec = [self.cfg['data']['train_chunklen_sec'], self.cfg['data']['test_chunklen_sec']]
        hoplen_sec = [self.cfg['data']['train_hoplen_sec'], self.cfg['data']['test_hoplen_sec']]
        last_frame_always_padding = [False, True]

        for idx, indexes_path in enumerate(self.indexes_path_list):
            if indexes_path.is_file():
                # indexes_path.unlink()
                with open(indexes_path, 'r') as f:
                    indices = f.read()
                if self.args.dataset in indices:
                    sys.exit(print('indices of dataset {} have been already extracted!'.format(self.args.dataset)))
            audio_cnt = 0
            f = open(indexes_path, 'a')
            if self.data_type == 'feature':
                frames_per_prediction = int(self.dataset.label_resolution / (self.cfg['data']['hoplen'] / self.cfg['data']['sample_rate']))
                paths_list_absolute = [path for path in sorted(self.feature_h5_dir.glob('*.h5')) if not path.name.startswith('.')]
                paths_list_relative = [path.relative_to(path.parent.parent) for path in paths_list_absolute]
                chunklen = int(chunklen_sec[idx] / self.dataset.label_resolution * frames_per_prediction) 
                hoplen = int(hoplen_sec[idx] / self.dataset.label_resolution * frames_per_prediction) 
                iterator = tqdm(paths_list_absolute, total=len(paths_list_absolute), unit='it')
                for path in iterator:
                    fn = paths_list_relative[audio_cnt]
                    with h5py.File(path, 'r') as hf:
                        num_frames = hf['feature'][:].shape[1]
                    data = np.zeros((1, num_frames))
                    segmented_indexes, segmented_pad_width = _segment_index(data, chunklen, hoplen, last_frame_always_paddding=last_frame_always_padding[idx])
                    for segmented_pairs in list(zip(segmented_indexes, segmented_pad_width)):
                        f.write('{},{},{},{},{}\n'.format(fn, segmented_pairs[0][0], segmented_pairs[0][1],\
                                segmented_pairs[1][0], segmented_pairs[1][1]))
                    audio_cnt += 1
                    tqdm.write('{},{}'.format(audio_cnt, fn))
            else:
                chunklen = int(chunklen_sec[idx] * self.cfg['data']['sample_rate'])     
                hoplen = int(hoplen_sec[idx] * self.cfg['data']['sample_rate'])
                paths_list_absolute = [path for path in sorted(self.wav_h5_dir_list[0].glob('*.h5')) if not path.name.startswith('.')]
                paths_list_relative = [path.relative_to(path.parent.parent) for path in paths_list_absolute]
                iterator = tqdm(paths_list_absolute, total=len(paths_list_absolute), unit='it')
                for path in iterator:
                    fn = paths_list_relative[audio_cnt]
                    with h5py.File(path, 'r') as hf:
                        data_length = hf['waveform'][:].shape[1]
                    data = np.zeros((1, data_length))
                    segmented_indexes, segmented_pad_width = _segment_index(data, chunklen, hoplen, last_frame_always_paddding=last_frame_always_padding[idx])
                    for segmented_pairs in list(zip(segmented_indexes, segmented_pad_width)):
                        f.write('{},{},{},{},{}\n'.format(fn, segmented_pairs[0][0], segmented_pairs[0][1],\
                                segmented_pairs[1][0], segmented_pairs[1][1]))
                    audio_cnt += 1
                    tqdm.write('{},{}'.format(audio_cnt, fn))
            f.close()
     

    def extract_scalar(self):
        """Extract scalar of features for normalization

        """
        if self.scalar_path.is_file():
            sys.exit('{} exists!'.format(self.scalar_path))
        if self.data_type == 'wav':
            self.extract_scalar_data()
    

    def extract_scalar_data(self):
        """ Extract scalar and store to hdf5 file

        """
        print('Extracting scalar......\n')
        self.scalar_h5_dir.mkdir(parents=True, exist_ok=True)
        cuda_enabled = not self.args.no_cuda and torch.cuda.is_available()
        train_set = BaseDataset(self.args, self.cfg, self.dataset)
        data_generator = DataLoader(
            dataset=train_set,
            batch_size=32,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        af_extractor = get_afextractor(self.cfg, cuda_enabled).eval()
        iterator = tqdm(enumerate(data_generator), total=len(data_generator), unit='it')
        scalar_list = [preprocessing.StandardScaler() for _ in range(self.channels_dict[self.cfg['data']['audio_feature']])]
        begin_time = timer()
        for it, batch_sample in iterator:
            if it == len(data_generator):
                break
            batch_x = batch_sample['waveform'][:]
            batch_x.require_grad = False
            if cuda_enabled:
                batch_x = batch_x.cuda(non_blocking=True)
            batch_y = af_extractor(batch_x).transpose(0, 1) # (C,N,T,F)
            C, _, _, F = batch_y.shape
            batch_y = batch_y.reshape(C, -1, F).cpu().numpy()
            for i_channel in range(len(scalar_list)):
                scalar_list[i_channel].partial_fit(batch_y[i_channel])
        iterator.close()
        mean = []
        std = []
        for i_chan in range(len(scalar_list)):
            mean.append(scalar_list[i_chan].mean_)
            std.append(np.sqrt(scalar_list[i_chan].var_))
        mean = np.stack(mean)[None, :, None, :]
        std = np.stack(std)[None, :, None, :]

        # save to h5py
        with h5py.File(self.scalar_path, 'w') as hf:
            hf.create_dataset(name='mean', data=mean, dtype=np.float32)
            hf.create_dataset(name='std', data=std, dtype=np.float32)
        print('Mean shape: ', mean.shape, '  Std shape: ', std.shape)
        print("\nScalar saved to {}\n".format(str(self.scalar_path)))
        print("Extacting scalar finished! Time spent: {:.3f} s\n".format(timer() - begin_time))           

