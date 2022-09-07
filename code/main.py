import sys
from timeit import default_timer as timer
from utils.cli_parser import parse_cli_overides
from utils.config import get_dataset
from learning.preprocess import Preprocess
from utils.ddp_init import cleanup, spawn_nproc, setup
import torch
from utils.common import prepare_train_id
from learning import initialize, train, infer

feature_type = {
    'mic': ['salsalite', 'logmelgcc', 'logmel'],
    'foa': ['salsa', 'logmelIV', 'logmel']
}

def training(rank, args, cfg, dataset):
    # Init DDP
    setup(rank=rank, args=args)
    train_initializer = initialize.init_train(args, cfg, dataset)
    train.train(cfg, **train_initializer)

def main(args, cfg):
    """Execute a task based on the given command-line arguments.

    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, infer predictions, and
    evaluate predictions using the command-line interface.

    Args:
        args: command line arguments.
        cfg: configurations.
    Return:
        0: successful termination
        'any nonzero value': abnormal termination
    """

    assert cfg['data']['audio_feature'] in feature_type[cfg['data']['type']], \
        '{} is not the feature of {} signals.'.format(cfg['data']['audio_feature'], cfg['data']['type'])
    
    # Dataset initialization
    dataset = get_dataset(root_dir=cfg['dataset_dir'], cfg=cfg, args=args)

    # Preprocess
    if args.mode == 'preprocess':
        preprocessor = Preprocess(args, cfg, dataset)
        if args.preproc_mode == 'extract_data':
            preprocessor.extract_data()
        elif args.preproc_mode == 'extract_mic_features':
            preprocessor.extract_mic_features()
        elif args.preproc_mode == 'extract_pit_label':
            preprocessor.extract_PIT_label()
        elif args.preproc_mode == 'extract_indexes':
            preprocessor.extract_index()
        elif args.preproc_mode == 'extract_scalar':
            preprocessor.extract_scalar()
        elif args.preproc_mode == 'extract_adpit_label':
            preprocessor.extract_ADPIT_label()

    # Train
    if args.mode == 'train':    
        prepare_train_id(args, cfg)
        spawn_nproc(training, args, cfg, dataset)

    # Inference
    elif args.mode == 'infer':
        infer_initializer = initialize.init_infer(args, cfg, dataset)
        infer.infer(cfg, dataset, **infer_initializer)

    
if __name__ == '__main__':
    args, cfg = parse_cli_overides()
    sys.exit(main(args, cfg))
