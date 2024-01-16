import os, gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

class cite_lgb:
    def __init__(self,params):
        self.params = params
     
        
        
    def correlation_score(self, y_true, y_pred):
        """Scores the predictions according to the competition rules. 
        It is assumed that the predictions are not constant.
        Returns the average of each sample's Pearson correlation coefficient"""
        if type(y_true) == pd.DataFrame: y_true = y_true.values
        if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
        corrsum = 0
        for i in range(len(y_true)):
            corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
        return corrsum / len(y_true)
        
    def predict(self,cite_train_x,cite_train_y,cite_test_x,meta):
       
        
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

            model = MultiOutputRegressor(lgb.LGBMRegressor(**self.params))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            corrscore = self.correlation_score(y_val, y_pred)
            test_pred = test_pred + model.predict(cite_test_x)
        return test_pred  

