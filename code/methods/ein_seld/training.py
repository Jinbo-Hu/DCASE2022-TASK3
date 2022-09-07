from pathlib import Path
import random
import sys
from timeit import default_timer as timer

import h5py
import numpy as np
import torch
from methods.training import BaseTrainer
from utils.ddp_init import reduce_value, gather_value, get_rank, get_world_size
from methods.utils.data_utilities import track_to_dcase_format, to_metrics_format

class Trainer(BaseTrainer):

    def __init__(self, args, cfg, dataset, af_extractor, valid_set, model, optimizer, losses, metrics):

        super().__init__()
        self.cfg = cfg
        self.af_extractor = af_extractor
        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.metrics = metrics
        self.cuda = args.cuda
        self.max_ov = dataset.max_ov
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.label_resolution = dataset.label_resolution

        # Load ground truth for dcase metrics
        self.valid_paths_dict = valid_set.paths_dict
        self.gt_metrics_dict = valid_set.gt_metrics_dict
        self.points_per_predictions = valid_set.points_per_predictions

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
            if args.cuda:
                self.mean = torch.tensor(self.mean, dtype=torch.float32).to(self.rank)
                self.std = torch.tensor(self.std, dtype=torch.float32).to(self.rank)

        self.init_train_losses()
    
    def init_train_losses(self):
        """ Initialize train losses

        """
        self.train_losses = {
            'loss_all': 0.,
            'loss_sed': 0.,
            'loss_doa': 0.,
        }

    def train_step(self, batch_sample, epoch_it):
        """ Perform a train step

        """
        
        batch_x = batch_sample['data']
        batch_target = {
            'sed': batch_sample['sed_label'],
            'doa': batch_sample['doa_label'],
            'ov': batch_sample['ov'],
        }
        
        if self.cuda:
            batch_x = batch_x.to(self.rank, non_blocking=True)
            batch_target['sed'] = batch_target['sed'].to(self.rank, non_blocking=True)
            batch_target['doa'] = batch_target['doa'].to(self.rank, non_blocking=True)
        
        self.optimizer.zero_grad()
        if self.af_extractor:
            self.af_extractor.train()
            batch_x = self.af_extractor(batch_x)
        self.model.train()

        if self.scalar_path.is_file():
            batch_x = (batch_x - self.mean) / self.std
        
        pred = self.model(batch_x)
        loss_dict = self.losses.calculate(pred, batch_target)
        loss_dict[self.cfg['training']['loss_type']].backward()
        self.optimizer.step()

        self.train_losses['loss_all'] += loss_dict['all'].detach()
        self.train_losses['loss_sed'] += loss_dict['sed'].detach()
        self.train_losses['loss_doa'] += loss_dict['doa'].detach()
        

    def validate_step(self, generator=None, max_batch_num=None, valid_type='train', epoch_it=0):
        """ Perform the validation on the train, valid set

        Generate a batch of segmentations each time
        """
        if valid_type == 'train':
            train_losses = self.train_losses.copy()
            self.init_train_losses()
            return train_losses

        elif valid_type == 'valid':
            pred_sed_list, pred_doa_list = [], []
            loss_all, loss_sed, loss_doa = 0., 0., 0.

            for batch_idx, batch_sample in enumerate(generator):
                if batch_idx == max_batch_num:
                    break
                batch_x = batch_sample['data']
                batch_target = {
                    'sed': batch_sample['sed_label'],
                    'doa': batch_sample['doa_label'],
                }
                if self.cuda:
                    batch_x = batch_x.to(self.rank, non_blocking=True)
                    batch_target['sed'] = batch_target['sed'].to(self.rank, non_blocking=True)
                    batch_target['doa'] = batch_target['doa'].to(self.rank, non_blocking=True)

                with torch.no_grad():
                    if self.af_extractor:
                        self.af_extractor.eval()
                        batch_x = self.af_extractor(batch_x)
                    self.model.eval()
                    if self.scalar_path.is_file():
                        batch_x = (batch_x - self.mean) / self.std
                    pred = self.model(batch_x)
                loss_dict = self.losses.calculate(pred, batch_target, epoch_it)

                pred['sed'] = torch.sigmoid(pred['sed'])
                loss_all += loss_dict['all'].detach()
                loss_sed += loss_dict['sed'].detach()
                loss_doa += loss_dict['doa'].detach()
                pred_sed_list.append(pred['sed'].detach())
                pred_doa_list.append(pred['doa'].detach())
            pred_sed = torch.concat(pred_sed_list, axis=0)
            pred_doa = torch.concat(pred_doa_list, axis=0)

            # gather data
            pred_sed = gather_value(pred_sed).cpu().numpy()
            pred_doa = gather_value(pred_doa).cpu().numpy()

            pred_sed_max = pred_sed.max(axis=-1)
            pred_sed_max_idx = pred_sed.argmax(axis=-1)
            pred_sed = np.zeros_like(pred_sed)
            for b_idx in range(pred_sed.shape[0]):
                for t_idx in range(pred_sed.shape[1]):
                    for track_idx in range(self.max_ov):
                        pred_sed[b_idx, t_idx, track_idx, pred_sed_max_idx[b_idx, t_idx, track_idx]] = \
                            pred_sed_max[b_idx, t_idx, track_idx]
            pred_sed = (pred_sed > self.cfg['training']['threshold_sed']).astype(np.float32)
            pred_sed = pred_sed.reshape(pred_sed.shape[0] * pred_sed.shape[1], self.max_ov, -1)
            pred_doa = pred_doa.reshape(pred_doa.shape[0] * pred_doa.shape[1], self.max_ov, -1)
            
            # convert Catesian to Spherical
            azi = np.arctan2(pred_doa[..., 1], pred_doa[..., 0])
            elev = np.arctan2(pred_doa[..., 2], np.sqrt(pred_doa[..., 0]**2 + pred_doa[..., 1]**2))
            pred_doa = np.stack((azi, elev), axis=-1) # (N, tracks, (azi, elev))
   
            frame_ind = 0
            for _, path in enumerate(self.valid_paths_dict):
                loc_frames = self.valid_paths_dict[path]
                num_frames = int(np.ceil(loc_frames / (self.cfg['data']['test_chunklen_sec'] / self.label_resolution)) *\
                    (self.cfg['data']['test_chunklen_sec'] / self.label_resolution))
                pred_dcase_format = track_to_dcase_format(pred_sed[frame_ind:frame_ind+loc_frames], pred_doa[frame_ind:frame_ind+loc_frames])
                pred_metrics_format = to_metrics_format(pred_dcase_format, num_frames=loc_frames)
                frame_ind += num_frames
                self.metrics.update(pred_metrics_format, self.gt_metrics_dict[path])
            
            out_losses = {
                'loss_all': loss_all / (batch_idx + 1),
                'loss_sed': loss_sed / (batch_idx + 1),
                'loss_doa': loss_doa / (batch_idx + 1),
            }
            for k, v in out_losses.items():
                out_losses[k] = reduce_value(v).cpu().numpy()  
            metrics_scores = self.metrics.calculate()

            return out_losses, metrics_scores

