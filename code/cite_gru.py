
#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd 
import gc
from tensorflow.keras.utils import plot_model

DATA_DIR = "../data/open-problems-multimodal"
feature_path = '../result/cite_fe_gru/'  
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')



# 合并特征为 GRU 做准备工作
def zscore(x):
    x_zscore = []
    for i in range(x.shape[0]):
        x_row = x[i]
        x_row = (x_row - np.mean(x_row)) / np.std(x_row)
        x_zscore.append(x_row)
    x_std = np.array(x_zscore)    
    return x_std

print('modify test id')
test_cite_inputs = pd.read_hdf(DATA_DIR+'test_cite_inputs.h5').reset_index()[['cell_id']]
test_cite_inputs_raw = pd.read_hdf(DATA_DIR+'test_cite_inputs_raw.h5').reset_index()

train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')

print('target')
train_cite_y = np.load(feature_path+'train_cite_targets.npy')    

print('cite_inputs_svd_clr')
cite_inputs_svd_clr = np.load(feature_path+'cite_inputs_svd_clr_200.npy')
train_cite_svd_clr = cite_inputs_svd_clr[:len(train_df)]
test_cite_svd_clr = cite_inputs_svd_clr[len(train_df):]
train_cite_svd_clr = zscore(train_cite_svd_clr)
test_cite_svd_clr = zscore(test_cite_svd_clr)

df_test_cite_svd_clr = pd.DataFrame(test_cite_svd_clr)
df_test_cite_svd_clr['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_svd_clr, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_svd_clr = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('cite_inputs_bio_norm')
cite_inputs_bio_norm_svd = np.load(feature_path+'cite_inputs_bio_norm_svd_100.npy')
train_cite_inputs_bio_norm_svd = cite_inputs_bio_norm_svd[:len(train_df)]
test_cite_inputs_bio_norm_svd = cite_inputs_bio_norm_svd[len(train_df):]
train_cite_inputs_bio_norm_svd = zscore(train_cite_inputs_bio_norm_svd)
test_cite_inputs_bio_norm_svd = zscore(test_cite_inputs_bio_norm_svd)

df_test_cite_inputs_bio_norm_svd = pd.DataFrame(test_cite_inputs_bio_norm_svd)
df_test_cite_inputs_bio_norm_svd['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_inputs_bio_norm_svd, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_inputs_bio_norm_svd = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('cite_inputs_raw_important_feats')
cite_inputs_feats = np.load(feature_path+'cite_inputs_raw_important_feats.npy') 
train_cite_inputs_feats = cite_inputs_feats[:len(train_df)]
test_cite_inputs_feats = cite_inputs_feats[len(train_df):]
train_cite_inputs_feats = zscore(train_cite_inputs_feats)
test_cite_inputs_feats = zscore(test_cite_inputs_feats)

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
train_cite_inputs_bio_norm_pca_64 = zscore(train_cite_inputs_bio_norm_pca_64)
test_cite_inputs_bio_norm_pca_64 = zscore(test_cite_inputs_bio_norm_pca_64)

df_test_cite_inputs_bio_norm_pca_64 = pd.DataFrame(test_cite_inputs_bio_norm_pca_64)
df_test_cite_inputs_bio_norm_pca_64['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_inputs_bio_norm_pca_64, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_inputs_bio_norm_pca_64 = test_cite_inputs_id.drop(['cell_id'],axis=1).values


print('lgb1_meta_features')
cite_lgb1_svd = np.load(feature_path+'cite_lgb1_svd_100.npy')
train_cite_lgb1_svd = cite_lgb1_svd[:len(train_df)]
test_cite_lgb1_svd = cite_lgb1_svd[len(train_df):]
train_cite_lgb1_svd = zscore(train_cite_lgb1_svd)
test_cite_lgb1_svd = zscore(test_cite_lgb1_svd)


print('lgb2_meta_features')
cite_lgb2_svd = np.load(feature_path+'cite_lgb2_svd_100.npy')
train_cite_lgb2_svd = cite_lgb2_svd[:len(train_df)]
test_cite_lgb2_svd = cite_lgb2_svd[len(train_df):]
train_cite_lgb2_svd = zscore(train_cite_lgb2_svd)
test_cite_lgb2_svd = zscore(test_cite_lgb2_svd)


print('lgb3_meta_features')
cite_lgb3_svd = np.load(feature_path+'cite_lgb3_svd_100.npy')
train_cite_lgb3_svd = cite_lgb3_svd[:len(train_df)]
test_cite_lgb3_svd = cite_lgb3_svd[len(train_df):]
train_cite_lgb3_svd = zscore(train_cite_lgb3_svd)
test_cite_lgb3_svd = zscore(test_cite_lgb3_svd)

df_test_cite_lgb3_svd = pd.DataFrame(test_cite_lgb3_svd)
df_test_cite_lgb3_svd['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_lgb3_svd, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_lgb3_svd = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('lgb4_meta_features')
cite_lgb4_svd = np.load(feature_path+'cite_lgb4_svd_100.npy')
train_cite_lgb4_svd = cite_lgb4_svd[:len(train_df)]
test_cite_lgb4_svd = cite_lgb4_svd[len(train_df):]
train_cite_lgb4_svd = zscore(train_cite_lgb4_svd)
test_cite_lgb4_svd = zscore(test_cite_lgb4_svd)

df_test_cite_lgb4_svd = pd.DataFrame(test_cite_lgb4_svd)
df_test_cite_lgb4_svd['cell_id'] = test_cite_inputs_raw['cell_id']
test_cite_inputs_id = test_cite_inputs.copy()
test_cite_inputs_id = test_cite_inputs_id.merge(df_test_cite_lgb4_svd, on=['cell_id'], how='left')
test_cite_inputs_id = test_cite_inputs_id.fillna(0)
test_cite_lgb4_svd = test_cite_inputs_id.drop(['cell_id'],axis=1).values

print('concatenate')
train_cite_X = np.concatenate([
                               train_cite_svd_clr,
                               train_cite_inputs_feats,
                               train_cite_inputs_bio_norm_svd,
                               train_cite_inputs_bio_norm_pca_64,
                               train_cite_lgb1_svd,
                               train_cite_lgb2_svd,
                               train_cite_lgb3_svd,
                               train_cite_lgb4_svd,
                                ],axis=1)

test_cite_X = np.concatenate([
                              test_cite_svd_clr,
                              test_cite_inputs_feats,
                              test_cite_inputs_bio_norm_svd, 
                              test_cite_inputs_bio_norm_pca_64,
                              test_cite_lgb1_svd,
                              test_cite_lgb2_svd,
                              test_cite_lgb3_svd,
                              test_cite_lgb4_svd,
                                ],axis=1)

np.save(feature_path+'train_cite_X.npy', train_cite_X)
np.save(feature_path+'test_cite_X.npy', test_cite_X)

train_cite_X = np.load(feature_path+'train_cite_X.npy')
test_cite_X = np.load(feature_path+'test_cite_X.npy')

train_cite_y = np.load(feature_path+'train_cite_targets.npy')  


# train_df.shape,test_df.shape,train_cite_X.shape,test_cite_X.shape,train_cite_y.shape



import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold

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



def cosine_similarity_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.math.l2_normalize(xm, axis = 1)
    t2_norm = tf.math.l2_normalize(ym, axis = 1)
    cosine = tf.keras.losses.CosineSimilarity(axis = 1)(t1_norm, t2_norm)
    return cosine

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
    
def nn_kfold(train_df, train_cite_X, train_cite_y, test_df, test_cite_X, network, folds, model_name):
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
        
        model = network(train_cite_X.shape[1])
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


def cite_cos_sim_model(len_num):
    
    
    input_num = tf.keras.Input(shape=(len_num))     
    x = input_num
    x0 =  tf.keras.layers.Reshape((1,x.shape[1]))(x)
    x0 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1800, activation='elu', kernel_initializer='Identity',return_sequences=False))(x0)
    x1 = tf.keras.layers.GaussianDropout(0.2)(x0)         
    x2 = tf.keras.layers.Dense(1800,activation ='elu',kernel_initializer='Identity',)(x1) 
    x3 = tf.keras.layers.GaussianDropout(0.2)(x2) 
    x4 = tf.keras.layers.Dense(1800,activation ='elu',kernel_initializer='Identity',)(x3) 
    x5 = tf.keras.layers.GaussianDropout(0.2)(x4)         
    x = tf.keras.layers.Concatenate()([
                       x1,x3,x5
                      ])
    output = tf.keras.layers.Dense(140, activation='linear')(x) 
    model = tf.keras.models.Model(input_num, output)
    lr=0.001
    adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, )
    model.compile(loss=cosine_similarity_loss, optimizer=adam,)
    model.summary()
    plot_model(model, to_file='model1.png', show_shapes=True)
    return model


BATCH_SIZE = 620
EPOCHS = 100
LR_FACTOR = 0.05
SEED = 666
N_FOLD = 5
folds = KFold(n_splits= N_FOLD, shuffle=True, random_state=SEED)     
oof_preds_cos,sub_preds_cos = nn_kfold(train_df, train_cite_X, train_cite_y,test_df, test_cite_X, cite_cos_sim_model, folds, 'cite_cos_model')



def cite_mse_model(len_num):
    
    #######################  svd  #######################   
    input_num = tf.keras.Input(shape=(len_num))     

    x = input_num
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x)    
    x = tf.keras.layers.GaussianDropout(0.1)(x)   
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x) 
    x = tf.keras.layers.GaussianDropout(0.1)(x)   
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x) 
    x = tf.keras.layers.GaussianDropout(0.1)(x)    
    x =  tf.keras.layers.Reshape((1,x.shape[1]))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(700, activation='swish',return_sequences=False))(x)
    x = tf.keras.layers.GaussianDropout(0.1)(x)  
    
    output = tf.keras.layers.Dense(140, activation='linear')(x) 

    model = tf.keras.models.Model(input_num, output)
    
    lr=0.0005
    weight_decay = 0.0001
    
    opt = tfa.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay
    )    

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,)
    model.summary()
    plot_model(model, to_file='model2.png', show_shapes=True)
    return model

BATCH_SIZE = 600
EPOCHS = 100
LR_FACTOR = 0.1
SEED = 666
folds = KFold(n_splits= 5, shuffle=True, random_state=SEED)    

# zscore for target
train_cite_y = zscore(train_cite_y)

oof_preds_mse,sub_preds_mse = nn_kfold(train_df, train_cite_X, train_cite_y, test_df, test_cite_X, cite_mse_model, folds, 'cite_mse_model')


oof_preds_cos = zscore(oof_preds_cos)
oof_preds_mse = zscore(oof_preds_mse)
oof_preds = oof_preds_cos*0.55 + oof_preds_mse*0.45
cv = correlation_score(train_cite_y,  oof_preds)
print ('Cite Blend:',cv)     

sub_preds_cos = zscore(sub_preds_cos)
sub_preds_mse = zscore(sub_preds_mse)
sub_preds = sub_preds_cos*0.55 + sub_preds_mse*0.45


del train_df,test_df,train_cite_X,test_cite_X,train_cite_y
del oof_preds_cos,oof_preds_mse,sub_preds_cos,sub_preds_mse
gc.collect()

submission = pd.read_csv( DATA_DIR +'/sample_submission.csv')   
submission.loc[:48663*140-1,'target'] = sub_preds.reshape(-1)
submission.to_csv(f'../result/cite_sub/GRU_submission.csv', index=False) 

