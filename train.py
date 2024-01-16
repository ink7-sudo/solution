
import os, gc
import pandas as pd
import numpy as np
import model.lgb as lgb 

DATA_DIR = "data/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"cite_day23_train.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"cite_day23_target.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"cite_day4_test.h5")

FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="citeseq"]
metadata_df.shape

conditions = [
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(4),
    ]

# create a list of the values we want to assign for each condition
values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# create a new column and use np.select to assign values to it using our lists as arguments
metadata_df['comb'] = np.select(conditions, values)

X = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
cell_index = X.index
meta = metadata_df.reindex(cell_index)
del X
gc.collect()
cite_train_x = pd.read_csv('result/cite_fe_lgb/X_876_day4.csv').values
cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
print(cite_train_x.shape)
print(cite_train_y.shape)
cite_test_x = pd.read_csv('result/cite_fe_lgb/Xt_876_day4.csv').values
cite_test_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values

params = {
    'n_estimators': 300, 
    'learning_rate': 0.1, 
    'max_depth': 10, 
    'num_leaves': 200,
    'min_child_samples': 250,
    'colsample_bytree': 0.8, 
    'subsample': 0.6, 
    "seed": 1,
    'verbosity':-1,
    "device": 'gpu',
    'max_bin': 64,
    "tree_type": "data",   
    "num_threads" :  32    
    }

lgb = lgb.cite_lgb(params)

predict = lgb.predict(cite_train_x,cite_train_y,cite_test_x,meta)
