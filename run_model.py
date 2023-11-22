"""
训练并评估单一模型
"""

import argparse

from mvts.pipeline import run_model
from mvts.utils import general_arguments, str2bool, str2float


def add_other_args(parser):
    for arg in general_arguments:
        if general_arguments[arg] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=None)
        elif general_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=None)
        elif general_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=None)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='multi_step',
                        help='the name of task, either multi_step or single_step')
    parser.add_argument('--model', type=str, default='AGCRN',
                        help='the name of model according to the task')
    parser.add_argument('--dataset', type=str, default='METR-LA',
                        help='the name of dataset')
    parser.add_argument('--config_file', type=str, default=None,
                        help='the file name of config file')
    parser.add_argument('--saved_model', type=bool, default=True,
                        help='whether save the trained model')
    parser.add_argument('--train', type=bool, default=True,
                        help='whether re-train model if the model is trained before')
    add_other_args(parser)
    return parser.parse_args()


def prepare_model_arguments(args):
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    
    model_arguments = {
        'task': args.task,
        'model_name': args.model,
        'dataset_name': args.dataset,
        'config_file': args.config_file,
        'saved_model': args.saved_model,
        'train': args.train,
        'other_args': other_args
    }

    return model_arguments


if __name__ == '__main__':
    args = parse_arguments()
    model_arguments = prepare_model_arguments(args)
    run_model(**model_arguments)

