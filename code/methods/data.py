from pathlib import Path
import os
import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.common import int16_samples_to_float32


class BaseDataset(Dataset):
    """ User defined datset

    """
    def __init__(self, args, cfg, dataset):
        """
        Args:
            args: input args
            cfg: configurations
            dataset: dataset used
        """
        super().__init__()

        self.args = args
        self.sample_rate = cfg['data']['sample_rate']
        self.data_type = 'wav' if cfg['data']['audio_feature'] in ['logmelIV', 'logmel'] else 'feature'
        
        # It's different from traing data
        # Chunklen and hoplen and segmentation. 
        hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset'])
        main_data_dir = hdf5_dir.joinpath('data').joinpath('{}fs'.format(cfg['data']['sample_rate'])).joinpath(self.data_type)
        
        if self.data_type == 'feature':
            self.data_dir = main_data_dir.joinpath('dev').joinpath(cfg['data']['audio_feature'])
            self.points_per_predictions = int(dataset.label_resolution / (cfg['data']['hoplen'] / cfg['data']['sample_rate']))
        else:
            self.data_dir = main_data_dir.joinpath('dev').joinpath(cfg['data']['type'])
            self.points_per_predictions = cfg['data']['sample_rate'] * dataset.label_resolution
        

        # Data path
        indexes_path = main_data_dir.joinpath('devset_{}sChunklen_{}sHoplen_train.csv'\
            .format(cfg['data']['train_chunklen_sec'], cfg['data']['train_hoplen_sec']))
        segments_indexes = pd.read_csv(indexes_path, header=None).values
        dataset_list = str(cfg['dataset_synth']).split(',')
        dataset_list.append('STARSS22')
        segments_indexes = [segment for segment in segments_indexes for _dataset in dataset_list if _dataset in segment[0]]
        self.segments_list = segments_indexes
        self.num_segments = len(self.segments_list)
        
    def __len__(self):
        """Get length of the dataset

        """
        return len(self.segments_list)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        clip_indexes = self.segments_list[idx]
        fn, segments = clip_indexes[0], clip_indexes[1:]
        data_path = self.data_dir.joinpath(fn)
        index_begin = segments[0]
        index_end = segments[1]
        pad_width_before = segments[2]
        pad_width_after = segments[3]
        if self.data_type == 'wav':
            with h5py.File(data_path, 'r') as hf:
                x = int16_samples_to_float32(hf['waveform'][:, index_begin: index_end]) 
            pad_width = ((0, 0), (pad_width_before, pad_width_after))
        else:
            with h5py.File(data_path, 'r') as hf:
                x = hf['feature'][:, index_begin: index_end] 
            pad_width = ((0, 0), (pad_width_before, pad_width_after), (0, 0))
        x = np.pad(x, pad_width, mode='constant')
        sample = {
            'waveform': x
        }
          
        return sample    


class PinMemCustomBatch:
    def __init__(self, batch_dict):
        batch_x = []
        for n in range(len(batch_dict)):
            batch_x.append(batch_dict[n]['waveform'])
        batch_x = np.stack(batch_x, axis=0)
        self.batch_out_dict = {
            'waveform': torch.tensor(batch_x, dtype=torch.float32),
        }

    def pin_memory(self):
        self.batch_out_dict['waveform'] = self.batch_out_dict['waveform'].pin_memory()
        return self.batch_out_dict


def collate_fn(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatch(batch_dict)
