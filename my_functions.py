from util.datasets import build_dataset, calculate_mean_std
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def set_mean_std(args):
    
    if args.stats_source == "custom":
            args.mean, args.std = calculate_mean_std(args.data_path + '/train/')
    else:
        args.mean, args.std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD