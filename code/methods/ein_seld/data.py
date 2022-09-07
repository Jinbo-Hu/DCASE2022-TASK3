from pathlib import Path
import pandas as pd
from timeit import default_timer as timer

import h5py
import numpy as np
import torch
from methods.utils.data_utilities import  load_output_format_file, to_metrics_format
from torch.utils.data import Dataset, Sampler
from utils.common import int16_samples_to_float32
from utils.ddp_init import get_rank, get_world_size

class UserDataset(Dataset):
    """ User defined datset

    """
    def __init__(self, cfg, dataset, dataset_type='train'):
        """
        Args:
            cfg: configurations
            dataset: dataset used
            dataset_type: 'train' | 'dev' | 'fold4_test' | 'eval_test' . 
                'train' and 'dev' are only used while training. 
                'fold4_test' and 'eval_test' are only used while infering.
        """
        super().__init__()

        self.cfg = cfg
        self.dataset_type = dataset_type
        self.label_resolution = dataset.label_resolution
        self.max_ov = dataset.max_ov
        self.num_classes = dataset.num_classes
        self.rank = get_rank()
        self.num_replicas = get_world_size()
        self.audio_feature = cfg['data']['audio_feature']
        self.data_type = 'wav' if cfg['data']['audio_feature'] == 'logmelIV' else 'feature'
        
        dataset_stage = 'eval' if 'eval' in dataset_type else 'dev'

        # data dir
        hdf5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset'])
        main_data_dir = hdf5_dir.joinpath('data').joinpath('{}fs'.format(cfg['data']['sample_rate']))\
            .joinpath(self.data_type)
        dataset_list = str(cfg['dataset_synth']).split(',')
        dataset_list.append('STARSS22')
        if self.data_type == 'feature':
            self.data_dir = main_data_dir.joinpath(dataset_stage).joinpath(cfg['data']['audio_feature'])
            self.points_per_predictions = int(dataset.label_resolution / (cfg['data']['hoplen'] / cfg['data']['sample_rate']))
        else:
            self.data_dir = main_data_dir.joinpath(dataset_stage).joinpath(cfg['data']['type'])
            self.points_per_predictions = cfg['data']['sample_rate'] * dataset.label_resolution
        
        # mete dir
        label_dir = hdf5_dir.joinpath('label')
        self.frame_meta_dir = label_dir.joinpath('frame')
        # self.track_meta_dir = label_dir.joinpath('track_pit_ov{}of5'.format(dataset.max_ov)).joinpath(dataset_stage)
        self.track_meta_dir = label_dir.joinpath('track_pit_ov{}of5_discontinuous'.format(dataset.max_ov)).joinpath(dataset_stage)

        # segments_list: data path and n_th segment  
        if self.dataset_type == 'train':
            indexes_path = main_data_dir.joinpath('{}set_{}sChunklen_{}sHoplen_train.csv'\
                .format(dataset_stage, cfg['data']['train_chunklen_sec'], cfg['data']['train_hoplen_sec']))
            segments_indexes = pd.read_csv(indexes_path, header=None).values
            train_fold = ['fold'+fold.strip() for fold in str(cfg['training']['train_fold']).split(',')]
            segments_indexes = [segment for segment in segments_indexes for _dataset in dataset_list if _dataset in segment[0]]
            self.segments_list = [clip_segment for clip_segment in segments_indexes \
                for fold in train_fold if fold in clip_segment[0] ]            

        elif self.dataset_type == 'dev':
            indexes_path = main_data_dir.joinpath('{}set_{}sChunklen_{}sHoplen_test.csv'\
                .format(dataset_stage, cfg['data']['test_chunklen_sec'], cfg['data']['test_hoplen_sec']))
            segments_indexes = pd.read_csv(indexes_path, header=None).values
            valid_fold = ['fold'+fold.strip() for fold in str(cfg['training']['valid_fold']).split(',')]
            segments_indexes = [segment for segment in segments_indexes for _dataset in dataset_list if _dataset in segment[0]]
            self.segments_list = [clip_segment for clip_segment in segments_indexes \
                for fold in valid_fold if fold in clip_segment[0] ]
            # load metadata
            self.paths_dict = {} # {path: num_frames}
            for segment in self.segments_list:
                self.paths_dict[segment[0]] = int(np.ceil(segment[2]/self.points_per_predictions))
            # each gpu use different sampler (the same as DistributedSampler)
            num_segments_per_gpu = int(np.ceil(len(self.segments_list) / self.num_replicas))
            self.segments_list = self.segments_list[self.rank * num_segments_per_gpu : (self.rank+1) * num_segments_per_gpu]
            self.gt_metrics_dict = {} # {path: metrics_dict}
            for file in self.paths_dict:
                path = self.frame_meta_dir.joinpath(str(file).replace('h5', 'csv'))
                valid_gt_dcaseformat = load_output_format_file(path)
                self.gt_metrics_dict[file] = to_metrics_format(label_dict=valid_gt_dcaseformat, \
                    num_frames=self.paths_dict[file], label_resolution=self.label_resolution)      

        elif 'test' in self.dataset_type:
            indexes_path = main_data_dir.joinpath('{}set_{}sChunklen_{}sHoplen_train.csv'\
                .format(dataset_stage, cfg['data']['test_chunklen_sec'], cfg['data']['test_hoplen_sec']))
            segments_indexes = pd.read_csv(indexes_path, header=None).values
            # test_fold = ['fold'+fold.strip() for fold in str(cfg['inference']['test_fold']).split(',')]
            test_fold = ['fold'+fold.strip() for fold in str(cfg['inference']['test_fold']).split(',')] \
                        if 'eval' not in dataset_type else ['mix']
            self.segments_list = [clip_segment for clip_segment in segments_indexes \
                for fold in test_fold if fold in clip_segment[0] ]
            self.paths_dict = {} # {path: num_frames}
            for segment in self.segments_list:
                self.paths_dict[segment[0]] = int(np.ceil(segment[2]/self.points_per_predictions))

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
        if 'test' not in self.dataset_type:
            meta_path = self.track_meta_dir.joinpath(fn)
            index_begin_label = int(index_begin / self.points_per_predictions)
            index_end_label = int(index_end / self.points_per_predictions)
            with h5py.File(meta_path, 'r') as hf:
                sed_label = hf['sed_label'][index_begin_label: index_end_label, ...]
                doa_label = hf['doa_label'][index_begin_label: index_end_label, ...]
            pad_width_after_label = int(self.cfg['data']['train_chunklen_sec'] / self.label_resolution - sed_label.shape[0])
            if pad_width_after_label != 0:
                sed_label_new = np.zeros((pad_width_after_label, self.max_ov, self.num_classes))
                sed_label = np.concatenate((sed_label, sed_label_new), axis=0)
                doa_label_new = np.zeros((pad_width_after_label, self.max_ov, 3))
                doa_label = np.concatenate((doa_label, doa_label_new), axis=0) 
        if 'test' not in self.dataset_type:
            sample = {
                'filename': fn,
                'data': x,
                'sed_label': sed_label,
                'doa_label': doa_label,
                'ov': str(max(np.sum(sed_label, axis=(1,2)).max(),1)),
            }
        else:
            sample = {
                'filename': fn,
                'data': x
            }
          
        return sample    


class UserBatchSampler(Sampler):
    """User defined batch sampler. Only for train set.

    """
    def __init__(self, clip_num, batch_size, seed=2022, drop_last=False):
        self.clip_num = clip_num
        self.batch_size = batch_size
        self.random_state = None
        self.indexes = np.arange(self.clip_num)
        self.pointer = 0
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = get_world_size()
        self.rank = get_rank()
        self.random_state = np.random.RandomState(self.seed+self.epoch)
        self.random_state.shuffle(self.indexes)
        if not self.drop_last:
            if self.clip_num % (self.batch_size*self.num_replicas) != 0:
                padding_size = self.batch_size*self.num_replicas - self.clip_num % (self.batch_size*self.num_replicas)
                self.indexes = np.append(self.indexes, self.indexes[:padding_size])
                self.clip_num = self.clip_num + padding_size

    def get_state(self):
        sampler_state = {
            'random': self.random_state.get_state(),
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return sampler_state

    def set_state(self, sampler_state):
        self.random_state.set_state(sampler_state['random'])
        self.indexes = sampler_state['indexes']
        self.pointer = sampler_state['pointer']
    
    def __iter__(self):
        """
        Return: 
            batch_indexes (int): indexes of batch
        """   
        while True:
            if self.pointer >= self.clip_num:
                self.pointer = 0
                self.random_state.shuffle(self.indexes)
            
            batch_indexes = self.indexes[self.pointer: self.pointer + self.batch_size * self.num_replicas]
            self.pointer += self.batch_size * self.num_replicas
            batch_indexes = batch_indexes[self.rank:self.clip_num:self.num_replicas]
            yield batch_indexes

    def __len__(self):
        return (self.clip_num + self.num_replicas * self.batch_size - 1) // (self.num_replicas * self.batch_size)


class PinMemCustomBatch:
    def __init__(self, batch_dict):
        batch_fn = []
        batch_x = []
        batch_ov = []
        batch_sed_label = []
        batch_doa_label = []
        
        for n in range(len(batch_dict)):
            batch_fn.append(batch_dict[n]['filename'])
            batch_x.append(batch_dict[n]['data'])
            batch_ov.append(batch_dict[n]['ov'])
            batch_sed_label.append(batch_dict[n]['sed_label'])
            batch_doa_label.append(batch_dict[n]['doa_label'])
            
        batch_x = np.stack(batch_x, axis=0)
        batch_sed_label = np.stack(batch_sed_label, axis=0)
        batch_doa_label = np.stack(batch_doa_label, axis=0)
        
        self.batch_out_dict = {
            'filename': batch_fn,
            'ov': batch_ov,
            'data': torch.tensor(batch_x, dtype=torch.float32),
            'sed_label': torch.tensor(batch_sed_label, dtype=torch.float32),
            'doa_label': torch.tensor(batch_doa_label, dtype=torch.float32),
        }

    def pin_memory(self):
        self.batch_out_dict['data'] = self.batch_out_dict['data'].pin_memory()
        self.batch_out_dict['sed_label'] = self.batch_out_dict['sed_label'].pin_memory()
        self.batch_out_dict['doa_label'] = self.batch_out_dict['doa_label'].pin_memory()
        return self.batch_out_dict


def collate_fn(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatch(batch_dict)


class PinMemCustomBatchTest:
    def __init__(self, batch_dict):
        batch_fn = []
        batch_x = []
        
        for n in range(len(batch_dict)):
            batch_fn.append(batch_dict[n]['filename'])
            batch_x.append(batch_dict[n]['data'])
        batch_x = np.stack(batch_x, axis=0)
        self.batch_out_dict = {
            'filename': batch_fn,
            'data': torch.tensor(batch_x, dtype=torch.float32)
        }

    def pin_memory(self):
        self.batch_out_dict['data'] = self.batch_out_dict['data'].pin_memory()
        return self.batch_out_dict


def collate_fn_test(batch_dict):
    """
    Merges a list of samples to form a mini-batch
    Pin memory for customized dataset
    """
    return PinMemCustomBatchTest(batch_dict)
