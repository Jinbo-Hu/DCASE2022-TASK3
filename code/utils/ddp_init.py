import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def spawn_nproc(demo_fn, args, cfg, dataset):
    mp.spawn(demo_fn,
            args=(args, cfg, dataset),
            nprocs=torch.cuda.device_count(),
            join=True)

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size=torch.cuda.device_count(), args=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def reduce_value(value, average=True):
    world_size = get_world_size()
    if not type(value) == torch.Tensor:
        value = torch.as_tensor(value).to(get_rank())
    if world_size > 1:  # single GPU
        dist.all_reduce(value)
        if average:
            value /= world_size
    return value

def gather_value(value):
    world_size = get_world_size()
    if not type(value) == torch.Tensor:
        value = torch.as_tensor(value).to(get_rank())
    if world_size > 1:  # more than 1 GPU
        value_list = [torch.zeros_like(value) for _ in range(world_size)]
        dist.all_gather(value_list, value)
        return torch.concat(value_list, dim=0)
    else:
        return value

def rank_barrier():
    rank = get_rank()
    if dist.is_initialized():
        dist.barrier(device_ids=[rank])