import os
import sys
import argparse

if os.path.basename(os.getcwd()) == 'examples':
    sys.path.append("../")

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader
from deepforest import main
from pytorch_lightning.loggers import CometLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepForest model on MillionTrees dataset")
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args

def convert_unknown_args_to_dict(unknown_args):
    kwargs = {}
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            kwargs[key] = None
        else:
            if kwargs[key] is None:
                kwargs[key] = arg
            else:
                kwargs[key] += f" {arg}"
    return kwargs


def main():
    args, unknown_args = parse_args()
    kwargs = convert_unknown_args_to_dict(unknown_args)

    dataset = get_dataset("TreeBoxes", root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
    train_loader = get_train_loader("standard", train_dataset, batch_size=2)

    m = main.deepforest(config_args=kwargs)

    # Load the pre-trained tree model
    m.load_model("Weecology/DeepForest-tree")

    # Create a trainer 
    comet_logger = CometLogger()

    m.create_trainer(logger=comet_logger)
    m.trainer.fit(m, train_loader)

if __name__ == "__main__":
    main()