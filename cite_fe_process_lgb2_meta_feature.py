import pandas as pd
import numpy as np
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import pickle
from sklearn.model_selection import StratifiedKFold,KFold
from scipy.sparse import hstack,vstack,csr_matrix,save_npz,load_npz
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
############################################################################
#----- work folder -----
############################################################################
settings ={"input_path":"../data/open-problems-multimodal/",
            "features_path":"../result/cite_fe_gru/"
            }

input_path = settings['input_path']
feature_path = settings['features_path']


# load clr svd,fine tuned process,correlated and important feaures

print('modify test')
test_cite_inputs = pd.read_hdf(input_path+'test_cite_inputs.h5').reset_index()[['cell_id']]
test_cite_inputs_raw = pd.read_hdf(input_path+'test_cite_inputs_raw.h5').reset_index()

print('target')
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')

train_cite_y = np.load(feature_path+'train_cite_targets.npy')    

print('cite_inputs_svd_clr')
cite_inputs_svd_clr = np.load(feature_path+'cite_inputs_svd_clr_200.npy')
train_cite_svd_clr = cite_inputs_svd_clr[:len(train_df)]
test_cite_svd_clr = cite_inputs_svd_clr[len(train_df):]

df_test_cite_svd_clr = pd.DataFrame(test_cite_svd_clr)
df_test_cite_svd_clr['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_svd_clr, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_svd_clr = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('cite_inputs_bio_norm_svd_100')
cite_inputs_bio_norm_2_svd = np.load(feature_path+'cite_inputs_bio_norm_svd_100.npy')
train_cite_inputs_bio_norm_2_svd = cite_inputs_bio_norm_2_svd[:len(train_df)]
test_cite_inputs_bio_norm_2_svd = cite_inputs_bio_norm_2_svd[len(train_df):]

df_test_cite_inputs_bio_norm_2_svd = pd.DataFrame(test_cite_inputs_bio_norm_2_svd)
df_test_cite_inputs_bio_norm_2_svd['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_inputs_bio_norm_2_svd, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_inputs_bio_norm_2_svd = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('cite_inputs_raw_important_feats')
cite_inputs_feats = np.load(feature_path+'cite_inputs_raw_important_feats.npy',allow_pickle=True)  
train_cite_inputs_feats = cite_inputs_feats[:len(train_df)]
test_cite_inputs_feats = cite_inputs_feats[len(train_df):]

df_test_cite_inputs_feats = pd.DataFrame(test_cite_inputs_feats)
df_test_cite_inputs_feats['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_inputs_feats, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_inputs_feats = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('cite_inputs_bio_norm_pca_64')
cite_inputs_bio_norm_pca_64 = np.load(feature_path+'cite_inputs_bio_norm_pca_64.npy')
train_cite_inputs_bio_norm_pca_64 = cite_inputs_bio_norm_pca_64[:len(train_df)]
test_cite_inputs_bio_norm_pca_64 = cite_inputs_bio_norm_pca_64[len(train_df):]

df_test_cite_inputs_bio_norm_pca_64 = pd.DataFrame(test_cite_inputs_bio_norm_pca_64)
df_test_cite_inputs_bio_norm_pca_64['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_inputs_bio_norm_pca_64, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_inputs_bio_norm_pca_64 = test_cite_inputs_id.drop(['cell_id'],axis=1).values


print('concatenate')
train_cite_X = np.concatenate([
                               train_cite_svd_clr,
                               train_cite_inputs_feats,
                               train_cite_inputs_bio_norm_2_svd,
                               train_cite_inputs_bio_norm_pca_64,
                                ],axis=1)

test_cite_X = np.concatenate([
                              test_cite_svd_clr,
                              test_cite_inputs_feats,
                              test_cite_inputs_bio_norm_2_svd,
                              test_cite_inputs_bio_norm_pca_64
                                ],axis=1)


# ====================================================
# lightgbm
# ====================================================
def lgb_kfold(train_df, test_df, train_cite_X, train_cite_y, test_cite_X, folds):
    params = {    
        'objective' : 'rmse',
        'metric' : 'mse', 
         'num_leaves': 128,
         'min_data_in_leaf': 30,
         'learning_rate': 0.01,
         'max_depth': 7,
         "boosting": "dart",
         "feature_fraction": 0.7,
         "bagging_freq": 1,
         "bagging_fraction": 0.7,
         "bagging_seed": 42,
         "lambda_l1":0.1,
         "lambda_l2":1,
         "verbosity": -1, 
         "device":'gpu' ,
         "max_bin":64  ,
         "tree_type": "data",   
          "num_threads" :  32          
                 }      
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=100)]
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)): 
        print ('n_fold:',n_fold)
        train_x = train_cite_X[train_idx]
        valid_x = train_cite_X[valid_idx]
        train_y = train_cite_y[train_idx]
        valid_y = train_cite_y[valid_idx]

        dtrain = lgb.Dataset(
            train_x, label=train_y,)
        dval = lgb.Dataset(
            valid_x, label=valid_y, reference=dtrain,)
        bst = lgb.train(
            params, dtrain, num_boost_round=5000,
            valid_sets=[dval],callbacks=callbacks
        )

        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        sub_preds += bst.predict(test_cite_X, num_iteration=bst.best_iteration) / folds.n_splits         
        
    return oof_preds,sub_preds


seed = 666
folds = KFold(n_splits= 3, shuffle=True, random_state=seed)  
train_preds = []
test_preds = []
for i in range(140):
    print('=====================')
    print(i)
    train_cite_y_single = train_cite_y[:,i]
    oof_preds,sub_preds = lgb_kfold(train_df, test_df, train_cite_X, train_cite_y_single, test_cite_X, folds)
    train_preds.append(oof_preds)
    test_preds.append(sub_preds)


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

oof_preds = np.zeros((len(train_df), 140))
for n in range(len(train_preds)):
    oof_preds[:,n] =  train_preds[n]

cv = correlation_score(train_cite_y, oof_preds)
print (cv)

sub_preds = np.zeros((len(test_df), 140))
for n in range(len(test_preds)):
    sub_preds[:,n] =  test_preds[n]  

lgb2 = np.concatenate([oof_preds,sub_preds],axis=0)

tsvd = TruncatedSVD(n_components=100, algorithm='arpack')
lgb2_svd = tsvd.fit_transform(lgb2)
np.save(feature_path+'cite_lgb2_svd_100.npy', lgb2_svd)

