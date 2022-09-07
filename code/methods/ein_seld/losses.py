import numpy as np
import torch
import sys
from methods.utils.loss_utilities import BCEWithLogitsLoss, MSELoss
from torch import linalg as LA
from itertools import permutations

class Losses:
    def __init__(self, cfg):
        self.cfg = cfg
        self.beta = cfg['training']['loss_beta']
        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]
        self.names = ['loss_all'] + [loss.name for loss in self.losses] 

    def calculate(self, pred, target, epoch_it=0):
        if 'PIT' not in self.cfg['training']['PIT_type']:
            loss_sed = self.losses[0].calculate_loss(pred['sed'], target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa = self.tPIT(pred, target)
        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
        losses_dict = {
            'all': loss_all,
            'sed': loss_sed,
            'doa': loss_doa,
        }
        return losses_dict    
    
    #### modify tracks
    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 6 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=3, num_classes], 
                'doa': [batch_size, T, num_tracks=3, doas=3]            
            }
        Return:
            loss_sed: Find a possible permutation to get the lowest loss of sed. 
            loss_doa: Find a possible permutation to get the lowest loss of doa. 
        """

        perm_list = list(permutations(range(pred['doa'].shape[2])))
        loss_sed_list = []
        loss_doa_list = []
        loss_list = []
        loss_sed = 0
        loss_doa = 0
        updated_target_doa = 0
        updated_target_sed = 0
        for idx, perm in enumerate(perm_list):
            loss_sed_list.append(self.losses_pit[0].calculate_loss(pred['sed'], target['sed'][:,:,list(perm),:])) 
            loss_doa_list.append(self.losses_pit[1].calculate_loss(pred['doa'], target['doa'][:,:,list(perm),:]))
            loss_list.append(loss_sed_list[idx]+loss_doa_list[idx])
        loss_list = torch.stack(loss_list, dim=0)
        loss_idx = torch.argmin(loss_list, dim=0)
        for idx, perm in enumerate(perm_list):
            loss_sed += loss_sed_list[idx] * (loss_idx == idx)
            loss_doa += loss_doa_list[idx] * (loss_idx == idx)
            updated_target_doa += target['doa'][:, :, list(perm), :] * ((loss_idx == idx)[:, :, None, None])
            updated_target_sed += target['sed'][:, :, list(perm), :] * ((loss_idx == idx)[:, :, None, None])
        loss_sed = loss_sed.mean()
        loss_doa = loss_doa.mean()
        updated_target = {
            'doa': updated_target_doa,
            'sed': updated_target_sed,
        }

        return loss_sed, loss_doa