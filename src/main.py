
import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='PDM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='sports', help='name of datasets')
    parser.add_argument('--mib', '-mib', type=float, default=1e-1, help='name of datasets')
    parser.add_argument('--beta', '-b', type=float, default=1e-3, help='name of datasets')

    args, _ = parser.parse_known_args()
    
    config_dict = {
        'gpu_id': 0,
        'beta': args.beta,
        'mib': args.mib
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


