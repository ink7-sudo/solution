
import os, gc
import pandas as pd
import numpy as np
import model.lgb as lgb 
import model.prompt_learner as prompt_learner
from model.prompt_learner import cite_cos_sim_model
from model.prompt_learner import cite_mse_model

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'
# DATA_DIR = "data/open-problems-multimodal/"
# FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

# FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"cite_day23_train.h5")
# FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"cite_day23_target.h5")
# FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"cite_day4_test.h5")

# FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
# metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
# metadata_df = metadata_df[metadata_df.technology=="citeseq"]
# metadata_df.shape

# conditions = [
#     metadata_df['donor'].eq(13176) & metadata_df['day'].eq(2),
#     metadata_df['donor'].eq(13176) & metadata_df['day'].eq(3),
#     metadata_df['donor'].eq(13176) & metadata_df['day'].eq(4),
#     metadata_df['donor'].eq(31800) & metadata_df['day'].eq(2),
#     metadata_df['donor'].eq(31800) & metadata_df['day'].eq(3),
#     metadata_df['donor'].eq(31800) & metadata_df['day'].eq(4),
#     metadata_df['donor'].eq(32606) & metadata_df['day'].eq(2),
#     metadata_df['donor'].eq(32606) & metadata_df['day'].eq(3),
#     metadata_df['donor'].eq(32606) & metadata_df['day'].eq(4),
#     ]

# # create a list of the values we want to assign for each condition
# values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# # create a new column and use np.select to assign values to it using our lists as arguments
# metadata_df['comb'] = np.select(conditions, values)

# X = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
# cell_index = X.index
# meta = metadata_df.reindex(cell_index)
# del X
# gc.collect()
# cite_train_x = pd.read_csv('result/cite_fe_lgb/X_876_day4.csv').values
# cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
# print(cite_train_x.shape)
# print(cite_train_y.shape)
# cite_test_x = pd.read_csv('result/cite_fe_lgb/Xt_876_day4.csv').values
# cite_test_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values

# params = {
#     'n_estimators': 300, 
#     'learning_rate': 0.1, 
#     'max_depth': 10, 
#     'num_leaves': 200,
#     'min_child_samples': 250,
#     'colsample_bytree': 0.8, 
#     'subsample': 0.6, 
#     "seed": 1,
#     'verbosity':-1,
#     "device": 'gpu',
#     'max_bin': 64,
#     "tree_type": "data",   
#     "num_threads" :  32    
#     }

# lgb = lgb.cite_lgb(params)

# predict = lgb.predict(cite_train_x,cite_train_y,cite_test_x,meta)

DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"
feature_path = '/home/dujunjia/bio/solution/result/cite_fe_gru/'  
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')
DATA_DIR = "../data/open-problems-multimodal/"

train_cite_X = np.load(feature_path+'prompt_train_cite_X.npy')
test_cite_X = np.load(feature_path+'prompt_test_cite_X.npy')

train_cite_y = np.load(feature_path+'day4_train_cite_targets.npy')  
cite_prompt = np.load(feature_path+'cite_prompt.npy')

def zscore(x):
    x_zscore = []
    for i in range(x.shape[0]):
        x_row = x[i]
        x_row = (x_row - np.mean(x_row)) / np.std(x_row)
        x_zscore.append(x_row)
    x_std = np.array(x_zscore)    
    return x_std
# d_model = 400 # 模型维度
# num_heads = 16  # 注意力头的数量
# ff_dim = 2    # Feedforward层的维度



# frozen_model = cite_cos_sim_model(1600)



# model = prompt_learner.TemporalPromptGeneratorModel(d_model, num_heads, ff_dim,frozen_model)

# model.compile(optimizer="adam", loss="mse")
# history = model.fit(train_cite_X,train_cite_y,
#                     batch_size=1600,
#                     epochs=420)
# predictions = model.predict(test_cite_X)

# def correlation_score(y_true, y_pred):
#     """Scores the predictions according to the competition rules. 
    
#     It is assumed that the predictions are not constant.
    
#     Returns the average of each sample's Pearson correlation coefficient"""
#     if type(y_true) == pd.DataFrame: y_true = y_true.values
#     if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
#     corrsum = 0
#     for i in range(len(y_true)):
#         corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
#     return corrsum / len(y_true)

# DATA_DIR = "../data/open-problems-multimodal/"


# FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
# cite_test_y = pd.read_hdf(FP_CITE_TEST_TARGETS).values

# print(correlation_score(cite_test_y,predictions))

# model.summary()


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

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_X, train_y, list_IDs, shuffle, batch_size, labels, ): 
        self.train_X = train_X
        self.train_y = train_y
        self.list_IDs = list_IDs        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.list_IDs) // self.batch_size
        return ct
    
    def __getitem__(self, idx):
        'Generate one batch of data'
        indexes = self.list_IDs[idx*self.batch_size:(idx+1)*self.batch_size]
    
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.labels: return X, y
        else: return X
 
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.list_IDs) )
        if self.shuffle: 
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'    
        X = self.train_X[list_IDs_temp]
        y = self.train_y[list_IDs_temp]        
        return X, y

    
def nn_kfold(train_df, train_cite_X, train_cite_y, test_df, test_cite_X, network, folds, model_name,d_model,num_heads,ff_dim,frozen_model):
    oof_preds = np.zeros((train_df.shape[0],140))
    sub_preds = np.zeros((test_df.shape[0],140))
    cv_corr = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df,)):          
        print (n_fold)
        train_x = train_cite_X[train_idx]
        valid_x = train_cite_X[valid_idx]
        train_y = train_cite_y[train_idx]
        valid_y = train_cite_y[valid_idx]

        train_x_index = train_df.iloc[train_idx].reset_index(drop=True).index
        valid_x_index = train_df.iloc[valid_idx].reset_index(drop=True).index

        model = network(d_model, num_heads, ff_dim,frozen_model)
        model.compile(optimizer="adam", loss="mse")
        #model = network(train_cite_X.shape[1])
        filepath = model_name+'_'+str(n_fold)+'.h5'
        es = tf.keras.callbacks.EarlyStopping(patience=10, mode='min', verbose=1) 
        checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True,save_weights_only=True,mode='min') 
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR, patience=6, verbose=1)
    
        train_dataset = DataGenerator(
            train_x,
            train_y,
            list_IDs=train_x_index, 
            shuffle=True, 
            batch_size=BATCH_SIZE, 
            labels=True,
        )
        
        valid_dataset = DataGenerator(
            valid_x,
            valid_y,
            list_IDs=valid_x_index, 
            shuffle=False, 
            batch_size=BATCH_SIZE, 
            labels=True,
        )
    
        hist = model.fit(train_dataset,
                        validation_data=valid_dataset,  
                        epochs=EPOCHS, 
                        callbacks=[checkpoint,es,reduce_lr_loss],
                        workers=4,
                        verbose=1)  
    
        model.load_weights(filepath)
        
        oof_preds[valid_idx] = model.predict(valid_x, 
                                batch_size=BATCH_SIZE,
                                verbose=1)
        
        oof_corr = correlation_score(valid_y,  oof_preds[valid_idx])
        cv_corr.append(oof_corr)
        print (cv_corr)       
        
        sub_preds += model.predict(test_cite_X, 
                                batch_size=BATCH_SIZE,
                                verbose=1) / folds.n_splits 
            
        del model
        gc.collect()
        tf.keras.backend.clear_session()    
    cv = correlation_score(train_cite_y,  oof_preds)
    print ('Overall:',cv)           
    return oof_preds,sub_preds    





BATCH_SIZE = 600
EPOCHS = 420
LR_FACTOR = 0.1
SEED = 666
folds = KFold(n_splits= 5, shuffle=True, random_state=SEED)    
d_model = 400 # 模型维度
num_heads = 16  # 注意力头的数量
ff_dim = 6    # Feedforward层的维度
frozen_model = cite_mse_model(1600)

# zscore for target
train_cite_y = zscore(train_cite_y)
oof_preds_mse,sub_preds_mse = nn_kfold(train_df, train_cite_X, train_cite_y, test_df, test_cite_X, prompt_learner.TemporalPromptGeneratorModel, folds, 'prompt_mse_model',d_model,num_heads,ff_dim,frozen_model)

BATCH_SIZE = 620
EPOCHS = 420
LR_FACTOR = 0.05
SEED = 666
N_FOLD = 5
folds = KFold(n_splits= N_FOLD, shuffle=True, random_state=SEED)     
d_model = 400 # 模型维度
num_heads = 16  # 注意力头的数量
ff_dim = 2    # Feedforward层的维度
frozen_model = cite_cos_sim_model(1600)

oof_preds_cos,sub_preds_cos = nn_kfold(train_df, train_cite_X, train_cite_y,test_df, test_cite_X, prompt_learner.TemporalPromptGeneratorModel, folds, 'prompt_cos_model',d_model,num_heads,ff_dim,frozen_model)


oof_preds_cos = zscore(oof_preds_cos)
oof_preds_mse = zscore(oof_preds_mse)
oof_preds = oof_preds_cos*0.55 + oof_preds_mse*0.45
cv = correlation_score(train_cite_y,  oof_preds)
print ('Cite Blend:',cv)     

sub_preds_cos = zscore(sub_preds_cos)
sub_preds_mse = zscore(sub_preds_mse)

DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"


FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
cite_test_y = pd.read_hdf(FP_CITE_TEST_TARGETS).values

all = []
for w in np.arange(0,1,0.01):
    sub_preds = w * sub_preds_cos + (1-w) * sub_preds_mse
    ensemble_auc = correlation_score(cite_test_y , sub_preds)
    all.append( ensemble_auc )
best_weight = np.argmax( all )/100


print(best_weight)

sub_preds = best_weight * sub_preds_cos + (1-best_weight) * sub_preds_mse

np.save("prompt_sub_preds.npy",sub_preds)
print(max(all))
del train_df,test_df,train_cite_X,test_cite_X,train_cite_y
del oof_preds_cos,oof_preds_mse,sub_preds_cos,sub_preds_mse
gc.collect()