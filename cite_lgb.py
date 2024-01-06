#!/usr/bin/env python
# coding: utf-8
import os, gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor


import lightgbm as lgb


DATA_DIR = "../data/open-problems-multimodal/open-problems-multimodal"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")




# !pip install --quiet tables
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


metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="citeseq"]
metadata_df.shape



conditions = [
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(27678) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(7)
    ]

# create a list of the values we want to assign for each condition
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# create a new column and use np.select to assign values to it using our lists as arguments
metadata_df['comb'] = np.select(conditions, values)



X = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
cell_index = X.index
meta = metadata_df.reindex(cell_index)
del X
gc.collect()


cite_train_x = pd.read_csv('../result/cite_fe_lgb/X_876.csv').values

cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values

print(cite_train_x.shape)
print(cite_train_y.shape)

cite_test_x = pd.read_csv('../result/cite_fe_lgb/Xt_876.csv').values

params = {
    'n_estimators': 300, 
    'learning_rate': 0.1, 
    'max_depth': 10, 
    'num_leaves': 200,
    'min_child_samples': 250,
    'colsample_bytree': 0.8, 
    'subsample': 0.6, 
    "seed": 1,
    }



test_pred = 0
N_SPLITS_ANN = len(meta['comb'].value_counts())
kf = GroupKFold(n_splits=N_SPLITS_ANN)
for fold, (idx_tr, idx_va) in enumerate(kf.split(cite_train_x, groups=meta.comb)):
    model = None
    gc.collect()
    
    X_train = cite_train_x[idx_tr] 
    y_train = cite_train_y[idx_tr]
    X_val = cite_train_x[idx_va]
    y_val = cite_train_y[idx_va]
    
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    corrscore = correlation_score(y_val, y_pred)
    
    print(mse)
    print(corrscore)
    test_pred = test_pred + model.predict(cite_test_x)        



submission = pd.read_csv( DATA_DIR +'/sample_submission.csv')   
submission.loc[:48663*140-1,'target'] = test_pred.reshape(-1)
submission.to_csv(f'../result/cite_sub/LGBM_submission.csv', index=False) 
