from dance.datasets.multimodality import ModalityPredictionDataset
import os
import os.path as osp
import argparse

import config



parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cite")
args = parser.parse_args()
subtask = args.task

if subtask == 'cite':
    os.system("kaggle competitions download -c open-problems-multimodal")
    os.system("unzip open-problems-multimodal.zip -d " + str(config.RAW_DATA_DIR) + " open-problems-multimodal")
    os.system("rm -f open-problems-multimodal.zip")
    os.system("kaggle datasets download -d ryanholbrook/open-problems-raw-counts")
    os.system("unzip open-problems-raw-counts.zip -d " + str(config.RAW_DATA_DIR) + " open-problems-multimodal")
    os.system("rm -f open-problems-raw-counts.zip")

