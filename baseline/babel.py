
import argparse
import os
import random
import pandas as pd
import anndata
import mudata
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from dance import logger
from dance.data import Data
from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.babel import BabelWrapper
from dance.modules.multi_modality.predict_modality.scmm import MMVAE
from dance.utils import set_seed

settings ={"input_path":"../data/open-problems-multimodal/",
            "features_path":"/home/dujunjia/bio/solution/result/cite_fe_gru/",
            }

set_seed(42)
device = 'cuda'
input_path = settings['input_path']
feature_path = settings['features_path']
parser = argparse.ArgumentParser()

######## Important hyperparameters

parser.add_argument("--max_epochs", type=int, default=40)
parser.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate")
parser.add_argument("--batchsize", "-b", type=int, default=64, help="Batch size")
parser.add_argument("--hidden", type=int, default=64, help="Hidden dimensions")
parser.add_argument("--earlystop", type=int, default=2, help="Early stopping after N epochs")
parser.add_argument("--naive", "-n", action="store_true", help="Use a naive model instead of lego model")
parser.add_argument("--lossweight", type=float, default=1., help="Relative loss weight")
########

parser.add_argument("--model_folder", default="./")
parser.add_argument("--outdir", "-o", default="./", help="Directory to output to")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--device", default="cuda")
parser.add_argument("--cpus", default=1, type=int)
parser.add_argument("--rnd_seed", default=42, type=int)

args_defaults = parser.parse_args([])
args = argparse.Namespace(**vars(args_defaults))
args

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


train_cite_Y = pd.read_hdf(input_path+'cite_day23_target.h5').values
train_cite_X = pd.read_hdf(input_path+'cite_day23_train.h5').values
test_cite_X =  pd.read_hdf(input_path+'cite_day4_test.h5').values  
test_cite_Y =  pd.read_hdf(input_path+'cite_day4_target.h5').values

train_cite_X = torch.as_tensor(train_cite_X, dtype=None, device='cuda')
train_cite_Y = torch.as_tensor(train_cite_Y, dtype=None, device='cuda')
test_cite_X = torch.as_tensor(test_cite_X, dtype=None, device='cuda')
test_cite_Y = torch.as_tensor(test_cite_Y, dtype=None, device='cuda')

print(train_cite_X)
print(train_cite_Y)
model = BabelWrapper(args, dim_in=train_cite_X.shape[1], dim_out=train_cite_Y .shape[1])
model.fit(train_cite_X, train_cite_Y, val_ratio=0.15)
Y_predict = model.predict(test_cite_X )
print(model.score(test_cite_X , test_cite_Y))

print(correlation_score(test_cite_Y.cpu(),Y_predict.cpu()))

# model = MMVAE('rna-protein',args)
# model.fit(train_cite_X, train_cite_Y, val_ratio=0.15)
# Y_predict = model.predict(test_cite_X )
# print(model.score(test_cite_X , test_cite_Y))

# print(correlation_score(test_cite_Y.cpu(),Y_predict.cpu()))
