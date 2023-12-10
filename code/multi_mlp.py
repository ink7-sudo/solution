#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
import copy
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold



DATA_DIR = "/home/djj/bio/solution/data/open-problems-multimodal/open-problems-multimodal"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")



metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="multiome"]
metadata_df.shape




conditions = [
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(10),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(10),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(10),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(10),
    ]
# create a list of the values we want to assign for each condition
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# create a new column and use np.select to assign values to it using our lists as arguments
metadata_df['comb'] = np.select(conditions, values)



X = np.load('/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_idxcol.npz', allow_pickle=True)

cell_index = X['index']
meta = metadata_df.reindex(cell_index)



def NegativeCorrLoss(preds, targets):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    my = torch.mean(preds, dim=1)
    my = torch.tile(torch.unsqueeze(my, dim=1), (1, targets.shape[1]))

    ym = preds - my
    r_num = torch.sum(torch.multiply(targets, ym), dim=1)
    r_den = torch.sqrt(
        torch.sum(torch.square(ym), dim=1) * float(targets.shape[-1])
    )
    r = torch.mean(r_num / r_den)
    return -r

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


class CustomDataset(Dataset):
    def __init__(self, split, X_train, X_val, X_test, y_train, y_val):
        self.split = split

        if self.split == "train":
            self.data = X_train
            self.gt = y_train
        elif self.split == "val":
            self.data = X_val
            self.gt = y_val
        else:
            self.data = X_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx], self.gt[idx]
        elif self.split == "val":
            return self.data[idx], 0
        else:
            return self.data[idx]




def train_model(model, optimizer, dataloaders_dict,  true_test_mod2, scheduler, num_epochs):
    best_mse = 100
    best_model = 0
    best_cor = 0
    for epoch in range(num_epochs):
        y_pred = []

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, gts in tqdm(dataloaders_dict[phase]):
                inputs = inputs.cuda()
                gts = gts.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if phase == 'train':
                        loss = NegativeCorrLoss(outputs, gts)
                        running_loss += loss.item() * inputs.size(0)
                        loss.backward()
                        optimizer.step()
                    else:
                        y_pred.extend(outputs.cpu().numpy())


            if phase == "train":
                epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                y_pred = np.array(y_pred)
                cor = correlation_score(true_test_mod2, y_pred)
                print('cor: ', cor)
                if cor > best_cor:
                    best_model = copy.deepcopy(model)
                    torch.save(best_model,'multi_model.pth')
                    best_cor = cor
        scheduler.step(cor)
    print("Best cor: ", best_cor)
    
    return best_model




def infer(model, dataloader):
    y_pred = []
    model.eval()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.array(y_pred)

    
    return y_pred




class MultiNet(nn.Module): 
    def __init__(self, dim_mod1, dim_mod2):
        super(MultiNet, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 2048)
        self.fc = nn.Linear(2048, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.25)
        self.output = nn.Linear(512, dim_mod2)
    def forward(self, x):
        x = F.gelu(self.input_(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc(x))
        x = self.dropout2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout3(x)
        x = F.gelu(self.output(x))
        return x



X = pd.read_csv('/home/djj/bio/solution/result/multi_fe/X_164_l2.csv').values
Xt = pd.read_csv('/home/djj/bio/solution/result/multi_fe/Xt_164_l2.csv').values



Y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values
Y -= Y.mean(axis=1).reshape(-1, 1)
Y /= Y.std(axis=1).reshape(-1, 1)




X = X.astype('float32')
Xt = Xt.astype('float32')





y_pred = 0
N_SPLITS_ANN = len(meta['comb'].value_counts())
kf = GroupKFold(n_splits=N_SPLITS_ANN)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X, groups=meta.comb)):
        model = None
        gc.collect()
        X_train = X[idx_tr] 
        y_train = Y[idx_tr]
        X_val = X[idx_va]
        y_val = Y[idx_va]   
        X_test = Xt
        

        model = MultiNet(164,23418).cuda()

        lr = 1e-4
        
        optimizer_ft = optim.AdamW(model.parameters(), lr=lr)
        
        data = {x: CustomDataset(x, X_train, X_val, X_test, y_train, y_val) for x in ['train', 'val', 'test']}
        dataloaders_dict = {"train": torch.utils.data.DataLoader(data["train"], batch_size=512, shuffle=True, num_workers=8),
                        "val": torch.utils.data.DataLoader(data["val"], batch_size=512, shuffle=False, num_workers=8),
                        "test": torch.utils.data.DataLoader(data["test"], batch_size=512, shuffle=False, num_workers=8)}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', patience=30, factor=0.1, verbose=True)
        best_model_net = train_model(model, optimizer_ft, dataloaders_dict, y_val, scheduler, num_epochs=100)
        y_pred = y_pred + infer(best_model_net, dataloaders_dict["test"])



# Read the table of rows and columns required for submission
eval_ids = pd.read_parquet("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/evaluation.parquet")

# Convert the string columns to more efficient categorical types
eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())



# Prepare an empty series which will be filled with predictions
submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)
submission



y_columns = np.load("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_idxcol.npz",
                   allow_pickle=True)["columns"]

test_index = np.load("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz",
                    allow_pickle=True)["index"]


cell_dict = dict((k,v) for v,k in enumerate(test_index)) 
assert len(cell_dict)  == len(test_index)

gene_dict = dict((k,v) for v,k in enumerate(y_columns))
assert len(gene_dict) == len(y_columns)
eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)
submission.iloc[valid_multi_rows] = y_pred[eval_ids_cell_num[valid_multi_rows].to_numpy(),
eval_ids_gene_num[valid_multi_rows].to_numpy()]
del eval_ids_cell_num, eval_ids_gene_num, valid_multi_rows, eval_ids, test_index, y_columns
gc.collect()



submission.reset_index(drop=True, inplace=True)
submission.index.name = 'row_id'



submission.to_csv('/home/djj/bio/solution/result/multi_sub/MLP_submission.csv', index=False)

