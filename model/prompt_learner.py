import numpy as np 
import pandas as pd 
import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflow_addons as tfa
tf.config.run_functions_eagerly(True)
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Dense, Concatenate
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"
feature_path = '/home/dujunjia/bio/solution/result/cite_fe_gru/'  
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')
DATA_DIR = "../data/open-problems-multimodal/"


cite_prompt = np.load(feature_path+'cite_prompt.npy')


import torch
import torch.nn as nn
import torch.optim as optim



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
    lr=0.0005
    adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, )
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=adam,)
    plot_model(model, to_file='model1.png', show_shapes=True)
    return model



class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 定义整体模型
class TemporalPromptGeneratorModel(keras.Model):
    def __init__(self, d_model, num_heads, ff_dim, frozen_model):
        super(TemporalPromptGeneratorModel, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, ff_dim)
        self.frozen_model = frozen_model  

    def call(self, inputs, training):
        prompts = inputs[:, -1:, :]
        output = inputs[:1,-1:, :]
    
        for i in range(1,inputs.shape[0]):
            prompts = inputs[:i+1,-1:,:]
            encoder_output = self.encoder(prompts, training=training)
            output = tf.concat([output, encoder_output[-1:,:,:]], axis=0)

        updated_inputs = tf.concat([inputs, output], axis=1)
        batch_size = tf.shape(updated_inputs )[0]
        encoder_output = tf.reshape(updated_inputs, (batch_size, -1))
        frozen_output = self.frozen_model(encoder_output )
        return frozen_output

# 参数设置



train_cite_X = np.load(feature_path+'prompt_train_cite_X.npy')
test_cite_X = np.load(feature_path+'prompt_test_cite_X.npy')

train_cite_y = np.load(feature_path+'day4_train_cite_targets.npy')  
cite_prompt = np.load(feature_path+'cite_prompt.npy')





d_model = 400 # 模型维度
num_heads = 16  # 注意力头的数量
ff_dim = 2    # Feedforward层的维度



frozen_model = cite_cos_sim_model(1600)



model = TemporalPromptGeneratorModel(d_model, num_heads, ff_dim,frozen_model)

model.compile(optimizer="adam", loss="mse")
history = model.fit(train_cite_X,train_cite_y,
                    batch_size=64,
                    epochs=4)
predictions = model.predict(test_cite_X)

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

DATA_DIR = "../data/open-problems-multimodal/"


FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
cite_test_y = pd.read_hdf(FP_CITE_TEST_TARGETS).values

print(correlation_score(cite_test_y,predictions))

model.summary()