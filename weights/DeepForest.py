import argparse
import os
import numpy as np
import torch
from PIL import Image

from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader

from deepforest.main import deepforest
from pytorch_lightning.loggers import CometLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepForest model on MillionTrees dataset")
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args

def convert_unknown_args_to_dict(unknown_args):
    def set_nested_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    kwargs = {}
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg.lstrip('--').split('.')
            set_nested_dict(kwargs, key, None)
        else:
            set_nested_dict(kwargs, key, arg)
    return kwargs


def main():
    args, unknown_args = parse_args()
    kwargs = convert_unknown_args_to_dict(unknown_args)

    dataset = TreeBoxesDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/", geometry_name="boxes") 
    train_dataset = dataset.get_subset("train")
    train_loader = get_train_loader("standard", train_dataset, batch_size=2)

    m = deepforest(config_args=kwargs)
    m.config["train"]["csv_file"] ="<dummy file, existing dataloader>"

    # Load the pre-trained tree model
    m.load_model("Weecology/DeepForest-tree")

    # Create a trainer 
    comet_logger = CometLogger()

    m.create_trainer(logger=comet_logger, fast_dev_run=True)
    m.trainer.fit(m, train_loader)

    ## Evaluate the model
    box_dataset = get_dataset("TreeBoxes", root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    box_test_data = box_dataset.get_subset("test")
    test_loader = get_eval_loader("standard", box_test_data, batch_size=16)

    # Get predictions for the full test set
    all_y_pred = []
    all_y_true = []
    all_metadata = []

    for batch in test_loader:
        metadata, images, targets  = batch
        for image in images:
            y_pred = m.predict_tile(image=image.permute(1, 2, 0))
            all_y_pred.append(y_pred)
            all_y_true.append(targets)

    # Evaluate
    box_dataset.eval(all_y_pred, all_y_true, all_metadata)

if __name__ == "__main__":
    main()