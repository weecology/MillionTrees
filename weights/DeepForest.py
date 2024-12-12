import argparse
import os
import numpy as np
import torch
from PIL import Image
import pandas as pd

from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader

from deepforest.main import deepforest
from deepforest.visualize import plot_results
from deepforest.utilities import read_file
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
    box_test_data = box_dataset.get_subset("test",frac=0.1)

    test_loader = get_eval_loader("standard", box_test_data, batch_size=16)

    # Print the length of the test loader
    print("There are {} batches in the test loader".format(len(test_loader)))

    # Get predictions for the full test set
    all_y_pred = []
    all_y_true = []

    batch_index = 0
    for batch in test_loader:
        metadata, images, targets  = batch
        for image_metadata, image, image_targets in zip(metadata,images, targets):
            basename = box_dataset._filename_id_to_code[int(image_metadata[0])]
            # Deepforest likes 0-255 data, channels first
            channels_first = image.permute(1, 2, 0).numpy() * 255
            pred = m.predict_image(channels_first)
            pred.root_dir = os.path.join(box_dataset._data_dir._str, "images")
            if pred is None:
                y_pred = {}
                y_pred["y"] = torch.zeros(4)
                y_pred["labels"] = torch.zeros(1)
                y_pred["scores"] = torch.zeros(1)
            else:
                pred["image_path"] = basename
                # Reformat to million trees format
                y_pred = {}
                y_pred["y"] = torch.tensor(pred[["xmin", "ymin", "xmax","ymax"]].values.astype("float32"))
                y_pred["labels"] = torch.tensor(pred.label.apply(
                        lambda x: m.label_dict[x]).values.astype(np.int64))
                y_pred["scores"] = torch.tensor(pred.score.values.astype("float32"))

            if batch_index % 100 == 0:
                ground_truth = read_file(pd.DataFrame(image_targets["y"].numpy(),columns=["xmin","ymin","xmax","ymax"]))
                ground_truth["label"] = "Tree"
                plot_results(pred, ground_truth, image=channels_first.astype("int32"))
            
            all_y_pred.extend([y_pred])
            all_y_true.extend(targets)
            batch_index += 1

    # Evaluate
    box_dataset.eval(all_y_pred, all_y_true, box_test_data.metadata_array)

if __name__ == "__main__":
    main()