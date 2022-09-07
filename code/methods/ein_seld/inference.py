from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from methods.inference import BaseInferer
from methods.utils.data_utilities import *


class Inferer(BaseInferer):

    def __init__(self, cfg, dataset, af_extractor, model, cuda, test_set=None):
        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.cuda = cuda
        self.dataset = dataset
        self.paths_dict = test_set.paths_dict

        # Scalar
        cfg_data = cfg['data']
        dataset_name = '_'.join(sorted(str(cfg['dataset_synth']).split(',')))
        scalar_h5_dir = Path(cfg['hdf5_dir']).joinpath(cfg['dataset']).joinpath('data').\
            joinpath('{}fs'.format(cfg_data['sample_rate'])).joinpath('scalar')
        fn_scalar = '{}_nfft{}_hop{}_mel{}_{}.h5'.format(cfg['data']['audio_feature'], \
            cfg_data['nfft'], cfg_data['hoplen'], cfg_data['n_mels'], dataset_name)
        self.scalar_path = scalar_h5_dir.joinpath(fn_scalar)
        if self.scalar_path.is_file():
            print('scalar path is used!', self.scalar_path)
            with h5py.File(self.scalar_path, 'r') as hf:
                self.mean = hf['mean'][:]
                self.std = hf['std'][:]
            if self.cuda:
                self.mean = torch.tensor(self.mean, dtype=torch.float32).cuda(non_blocking=True)
                self.std = torch.tensor(self.std, dtype=torch.float32).cuda(non_blocking=True)

        self.label_resolution = dataset.label_resolution
        
    def infer(self, generator):
        pred_sed_list, pred_doa_list = [], []

        iterator = tqdm(generator)
        for batch_sample in iterator:
            batch_x = batch_sample['data']
            if self.cuda:
                batch_x = batch_x.cuda(non_blocking=True)
            with torch.no_grad():
                if self.af_extractor:
                    self.af_extractor.eval()
                    batch_x = self.af_extractor(batch_x)
                self.model.eval()
                if self.scalar_path.is_file():
                    batch_x = (batch_x - self.mean) / self.std
                pred = self.model(batch_x)
            
            pred['sed'] = torch.sigmoid(pred['sed'])
            
            pred_sed_list.append(pred['sed'].cpu().detach().numpy())
            pred_doa_list.append(pred['doa'].cpu().detach().numpy())
        
        iterator.close()
        pred_sed = np.concatenate(pred_sed_list, axis=0)
        pred_doa = np.concatenate(pred_doa_list, axis=0)

        pred_sed = pred_sed.reshape((pred_sed.shape[0] * pred_sed.shape[1], 3, -1))
        pred_doa = pred_doa.reshape((pred_doa.shape[0] * pred_doa.shape[1], 3, -1))
        pred = {
            'sed': pred_sed,
            'doa': pred_doa
        }
        return pred

    def fusion(self, submissions_dir, predictions_dir, preds):
        """ Average ensamble predictions

        """
        num_preds = len(preds)
        pred_sed = []
        pred_doa = []
        for n in range(num_preds):
            pred_sed.append(preds[n]['sed'])
            pred_doa.append(preds[n]['doa'])
        pred_sed = np.array(pred_sed).mean(axis=0) # Ensemble
        pred_doa = np.array(pred_doa).mean(axis=0) # Ensemble

        prediction_path = predictions_dir.joinpath('predictions.h5')
        with h5py.File(prediction_path, 'w') as hf:
            hf.create_dataset(name='sed', data=pred_sed, dtype=np.float32)
            hf.create_dataset(name='doa', data=pred_doa, dtype=np.float32)
        N = pred_sed.shape[0]
        pred_sed_max = pred_sed.max(axis=-1)
        pred_sed_max_idx = pred_sed.argmax(axis=-1)
        pred_sed = np.zeros_like(pred_sed)
        for b_idx in range(N):
            for track_idx in range(3):
                pred_sed[b_idx, track_idx, pred_sed_max_idx[b_idx, track_idx]] = \
                    pred_sed_max[b_idx, track_idx]
        pred_sed = (pred_sed > self.cfg['inference']['threshold_sed']).astype(np.float32)

        # convert Catesian to Spherical
        azi = np.arctan2(pred_doa[..., 1], pred_doa[..., 0])
        elev = np.arctan2(pred_doa[..., 2], np.sqrt(pred_doa[..., 0]**2 + pred_doa[..., 1]**2))
        pred_doa = np.stack((azi, elev), axis=-1) # (N, tracks, (azi, elev))
        frame_ind = 0
        for idx, path in enumerate(self.paths_dict):
            loc_frames = self.paths_dict[path]
            fn = path.split('/')[-1].replace('h5','csv')
            num_frames = int(np.ceil(loc_frames / (self.cfg['data']['test_chunklen_sec'] / self.label_resolution)) *\
                (self.cfg['data']['test_chunklen_sec'] / self.label_resolution))
            pred_dcase_format = track_to_dcase_format(pred_sed[frame_ind:frame_ind+loc_frames], pred_doa[frame_ind:frame_ind+loc_frames])
            csv_path = submissions_dir.joinpath(fn)
            write_output_format_file(csv_path, pred_dcase_format)
            frame_ind += num_frames
        
        print('Rsults are saved to {}\n'.format(str(submissions_dir)))


