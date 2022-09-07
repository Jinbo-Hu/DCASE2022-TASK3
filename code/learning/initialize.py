import logging
import random
import shutil
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils.common import create_logging
from utils.config import get_generator, store_config, get_losses, get_afextractor, get_models, get_optimizer, get_metrics, get_trainer
from utils.ddp_init import get_rank, get_world_size, rank_barrier
from learning.checkpoint import CheckpointIO


def init_train(args, cfg, dataset):
    """ Training initialization.

    Including Data generator, model, optimizer initialization.
    """

    train_initializer = None
    '''Cuda'''
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    rank = get_rank()
    world_size = get_world_size()
    
    ''' Reproducible seed set'''
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True  # Using random seed to fix the algorithm
    cudnn.benchmark = True    # Automatically find the most efficient algorithm for the current configuration. Set it False to reduce random

    '''Sharing directories'''
    out_train_dir = Path(cfg['workspace_dir']).joinpath('results').joinpath('out_train') \
            .joinpath(cfg['method']).joinpath(cfg['training']['train_id'])
    ckpts_dir = out_train_dir.joinpath('checkpoints')
    if rank == 0:
        print('Train ID is {}\n'.format(cfg['training']['train_id']))  
        out_train_dir.mkdir(parents=True, exist_ok=True)

    '''tensorboard and logging'''
    if rank == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tb_dir = out_train_dir.joinpath('tb').joinpath(current_time + '_' + socket.gethostname())
        tb_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = out_train_dir.joinpath('logs')
        create_logging(logs_dir, filemode='w')
        writer = SummaryWriter(log_dir=str(tb_dir))
        param_file = out_train_dir.joinpath('config.yaml')
        if param_file.is_file():
            param_file.unlink()
        store_config(param_file, cfg)
    rank_barrier()

    '''Data generator'''
    train_set, train_generator, batch_sampler = get_generator(args, cfg, dataset, generator_type='train')
    valid_set, valid_generator, _ = get_generator(args, cfg, dataset, generator_type='valid')
   
    '''Loss'''
    losses = get_losses(cfg)

    '''Metrics'''
    metrics = get_metrics(cfg, dataset) 

    '''Audio feature extractor'''
    af_extractor = get_afextractor(cfg, args.cuda)

    '''Model'''
    model = get_models(cfg, dataset, args.cuda) 
    if dist.is_initialized():
        torch.cuda.set_device(rank) # it's necessary, or CUDA_OUT_OF_MEMORY
        model = DDP(model, device_ids=[rank], output_device=rank)

    '''Optimizer'''
    optimizer = get_optimizer(cfg, af_extractor, model)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['lr_step_size'], 
                    gamma=cfg['training']['lr_gamma'])

    '''Trainer'''
    trainer = get_trainer(args=args, cfg=cfg, dataset=dataset, valid_set=valid_set,
         af_extractor=af_extractor, model=model, optimizer=optimizer, losses=losses, metrics=metrics)

    '''CheckpointIO'''
    if not cfg['training']['valid_fold']:
        metrics_names = losses.names
    else:
        metrics_names = metrics.names
    ckptIO = CheckpointIO(
        checkpoints_dir=ckpts_dir,
        model=model,
        optimizer=optimizer,
        batch_sampler=batch_sampler,
        metrics_names=metrics_names,
        num_checkpoints=1,  
        remark=cfg['training']['remark']
    )
    
    if cfg['training']['resume_model']:
        resume_path = ckpts_dir.joinpath(cfg['training']['resume_model'])
        logging.info('=====>> Resume from the checkpoint: {}......\n'.format(str(resume_path)))
        epoch_it, it = ckptIO.load(resume_path)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['training']['lr']
    else:
        epoch_it, it = 0, 0

    ''' logging and return '''
    logging.info('Train folds are: {}\n'.format(cfg['training']['train_fold']))
    logging.info('Valid folds are: {}\n'.format(cfg['training']['valid_fold']))
    logging.info('Training clip number is: {}\n'.format(len(train_set)))
    logging.info('Number of batches per epoch is: {}\n'.format(len(batch_sampler)))
    logging.info('Validation clip number is: {}\n'.format(len(valid_set) * world_size))

    train_initializer = {
        'writer': writer if rank == 0 else None,
        'train_generator': train_generator,
        'valid_generator': valid_generator,
        'lr_scheduler': lr_scheduler,
        'trainer': trainer,
        'ckptIO': ckptIO ,
        'epoch_it': epoch_it,
        'it': it
    }

    return train_initializer


def init_infer(args, cfg, dataset):
    """ Inference initialization.

    Including Data generator, model, optimizer initialization.
    """

    ''' Cuda '''
    args.cuda = not args.no_cuda and torch.cuda.is_available() 

    ''' Directories '''
    print('Inference ID is {}\n'.format(cfg['inference']['infer_id']))
    out_infer_dir = Path(cfg['workspace_dir']).joinpath('results').joinpath('out_infer')\
        .joinpath(cfg['method']).joinpath(cfg['inference']['infer_id'])
    if out_infer_dir.is_dir():
        shutil.rmtree(str(out_infer_dir))
    submissions_dir = out_infer_dir.joinpath('submissions')
    predictions_dir = out_infer_dir.joinpath('predictions')
    submissions_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    train_ids = [train_id.strip() for train_id in str(cfg['inference']['train_ids']).split(',')]
    models = [model.strip() for model in str(cfg['inference']['models']).split(',')]
    ckpts_paths_list = []
    ckpts_models_list = []
    for train_id, model_name in zip(train_ids, models):
        ckpts_dir = Path(cfg['workspace_dir']).joinpath('results').joinpath('out_train').joinpath(cfg['method'])\
            .joinpath(train_id).joinpath('checkpoints')
        ckpt_path = [path for path in sorted(ckpts_dir.iterdir()) if cfg['inference']['model_mark'] in path.stem]
        print('ckpt_name: ', ckpt_path, 'model_name: ', model_name)
        # ckpt_path = [path for path in sorted(ckpts_dir.iterdir()) if path.stem.split('_')[-1].isnumeric()] # 
        for path in ckpt_path:
            ckpts_paths_list.append(path)
            ckpts_models_list.append(model_name)
    
    ''' Parameters '''
    param_file = out_infer_dir.joinpath('config.yaml')
    if param_file.is_file():
        param_file.unlink()
    store_config(param_file, cfg)

    ''' Data generator '''
    test_set, test_generator, _ = get_generator(args, cfg, dataset, generator_type='test')

    ''' logging and return '''
    logging.info('Test clip number is: {}\n'.format(len(test_set)))

    infer_initializer = {
        'submissions_dir': submissions_dir,
        'predictions_dir': predictions_dir,
        'ckpts_paths_list': ckpts_paths_list,
        'ckpts_models_list': ckpts_models_list,
        'test_generator': test_generator,
        'cuda': args.cuda,
        'test_set': test_set
    }

    return infer_initializer
