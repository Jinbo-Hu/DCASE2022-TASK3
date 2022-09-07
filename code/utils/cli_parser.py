import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML
from termcolor import cprint


def parse_cli_overides():
    """Parse the command-line arguments.

    Parse args from CLI and override config dictionary entries

    This function implements the command-line interface of the program.
    The interface accepts general command-line arguments as well as
    arguments that are specific to a sub-command. The sub-commands are
    *preprocess*, *train*, *predict*, and *evaluate*. Specifying a
    sub-command is required, as it specifies the task that the program
    should carry out.

    Returns:
        args: The parsed arguments.
    """
    # Parse the command-line arguments, but separate the `--config_file`
    # option from the other arguments. This way, options can be parsed
    # from the config file(s) first and then overidden by the other
    # command-line arguments later.
    parser = argparse.ArgumentParser(
        description='Event Independent Network for DCASE2022.', 
        add_help=False
    )
    parser.add_argument('-c', '--config_file', default='./configs/ein_seld/seld.yaml', help='Specify config file', metavar='FILE')
    parser.add_argument('--dataset', default='STARSS22', type=str)
    subparsers = parser.add_subparsers(dest='mode')
    parser_preproc = subparsers.add_parser('preprocess')
    parser_train = subparsers.add_parser('train')
    parser_infer = subparsers.add_parser('infer')
    subparsers.add_parser('evaluate')

    # Require the user to specify a sub-command
    subparsers.required = True
    parser_preproc.add_argument('--preproc_mode', choices=['extract_data', 'extract_indexes', 'extract_pit_label',
                                 'extract_mic_features', 'extract_scalar', 'extract_adpit_label'], 
                                 required=True, help='select preprocessing mode')
    parser_preproc.add_argument('--dataset_type', default='dev', choices=['dev', 'eval'], 
                                help='select dataset to preprocess')
    parser_preproc.add_argument('--num_workers', type=int, default=8, metavar='N')
    parser_preproc.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser_train.add_argument('--seed', type=int, default=2022, metavar='N')
    parser_train.add_argument('--num_workers', type=int, default=8, metavar='N')
    parser_train.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser_train.add_argument('--port', type=int, default=12359,  metavar='N')
    parser_infer.add_argument('--num_workers', type=int, default=8, metavar='N')
    parser_infer.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

    args = parser.parse_args()
    args_dict = vars(args)
    cprint("Args:", "green")
    for key, value in args_dict.items():
        print(f"     {key:25s} -> {value}")

    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.default_flow_style = False
    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)
    cprint("Cfg:", "red")
    yaml.dump(cfg, sys.stdout, transform=replace_indent)

    return args, cfg

def replace_indent(stream):
    stream = "     " + stream
    return stream.replace("\n", "\n     ")
