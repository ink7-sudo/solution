#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd 
import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Dense, Concatenate

DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"
feature_path = '/home/dujunjia/bio/solution/result/cite_fe_gru/'  
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')
DATA_DIR = "../data/open-problems-multimodal/"


FP_CITE_TEST_TARGETS = os.path.join(DATA_DIR,"cite_day4_target.h5")
cite_test_y = pd.read_hdf(FP_CITE_TEST_TARGETS).values

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
    model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=adam,)
    plot_model(model, to_file='model1.png', show_shapes=True)
    return model

train_cite_X = np.load(feature_path+'train_cite_X.npy')
test_cite_X = np.load(feature_path+'test_cite_X.npy')






    