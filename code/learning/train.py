import logging
from timeit import default_timer as timer
from tqdm import tqdm
from utils.common import print_metrics
from utils.ddp_init import reduce_value, get_rank, get_world_size, rank_barrier

def train(cfg, **initializer):
    """Train

    """
    writer = initializer['writer']
    train_generator = initializer['train_generator']
    valid_generator = initializer['valid_generator']
    lr_scheduler = initializer['lr_scheduler']
    trainer = initializer['trainer']
    ckptIO = initializer['ckptIO']
    epoch_it = initializer['epoch_it']
    it = initializer['it']
    world_size = get_world_size()
    rank = get_rank()
    batchNum_per_epoch = len(train_generator)
    max_epoch = cfg['training']['max_epoch']
    
    if rank == 0:
        logging.info('===> Training mode\n')
    iterator = tqdm(train_generator, total=max_epoch*batchNum_per_epoch-it, unit='it')
    train_begin_time = timer()
    
    for batch_sample in iterator:
        epoch_it, rem_batch = it // batchNum_per_epoch, it % batchNum_per_epoch
        ################
        ## Validation
        ################
        if it % int(1*batchNum_per_epoch) == 0 :
            valid_begin_time = timer()
            train_time = valid_begin_time - train_begin_time
            train_losses = trainer.validate_step(valid_type='train', epoch_it=epoch_it)
            for k, v in train_losses.items():
                train_losses[k] = reduce_value(v).cpu().numpy()   / batchNum_per_epoch
            if not cfg['training']['valid_fold'] == 'none':
                valid_losses, valid_metrics = trainer.validate_step(
                    generator=valid_generator,
                    valid_type='valid',
                    epoch_it=epoch_it
                )
            valid_time = timer() - valid_begin_time
            if rank == 0:
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], it)
                logging.info('---------------------------------------------------------------------------------------------------'
                    +'------------------------------------------------------')
                logging.info('Iter: {},  Epoch/Total Epoch: {}/{},  Batch/Total Batch: {}/{}'.format(
                    it, epoch_it, max_epoch, rem_batch, batchNum_per_epoch))
                print_metrics(logging, writer, train_losses, it, set_type='train')
                if not cfg['training']['valid_fold'] == 'none':
                    print_metrics(logging, writer, valid_losses, it, set_type='valid')
                if not cfg['training']['valid_fold'] == 'none':
                    ####  SELD Metrics  ####
                    if cfg['method'] in ['ein_seld', 'multi_accdoa']:
                        print_metrics(logging, writer, valid_metrics['macro'], it, set_type='valid')
                        print_metrics(logging, writer, valid_metrics['micro'], it, set_type='valid')
                logging.info('Train time: {:.3f}s,  Valid time: {:.3f}s,  Lr: {}'.format(
                    train_time, valid_time, lr_scheduler.get_last_lr()[0]))
                if 'PIT_type' in cfg['training']:
                    logging.info('PIT type: {}'.format(cfg['training']['PIT_type']))
                logging.info('---------------------------------------------------------------------------------------------------'
                    +'------------------------------------------------------')
            rank_barrier()
         
            train_begin_time = timer()
        ###############
        ## Save model
        ###############
        # if rank == 0 :
        #     if rem_batch == 0 and it > 0:
        #         if not cfg['training']['valid_fold'] == 'none':
        #             if cfg['method'] in ['ein_seld', 'multi_accdoa']:
        #                 ckptIO.save(epoch_it, it, metrics=valid_metrics['macro'], key_rank='seld_macro', rank_order='low', max_epoch=max_epoch)
        #             else:
        #                 ckptIO.save(epoch_it, it, metrics=valid_losses, key_rank='loss_all', rank_order='low', max_epoch=max_epoch)
        #         else:
        #             ckptIO.save(epoch_it, it, metrics=train_losses, key_rank='loss_all', rank_order='low', max_epoch=max_epoch)
        # rank_barrier()
        ###############
        ## Finish training
        ###############
        if it == max_epoch * batchNum_per_epoch:
            iterator.close()
            break

        ###############
        ## Train
        ###############
        trainer.train_step(batch_sample, epoch_it)
        if rem_batch == 0 and it > 0:
            lr_scheduler.step()
            
        it += 1
        
    iterator.close()

