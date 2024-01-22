import numpy as np 
import pandas as pd 
import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.layers import Input, Dense, Concatenate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



DATA_DIR = "/home/dujunjia/bio/solution/data/open-problems-multimodal/"
feature_path = '/home/dujunjia/bio/solution/result/cite_fe_gru/'  
train_df = pd.read_feather(feature_path+'train_cite_inputs_id.feather')
test_df = pd.read_feather(feature_path+'test_cite_inputs_id.feather')
DATA_DIR = "../data/open-problems-multimodal/"


cite_prompt = np.load(feature_path+'cite_prompt.npy')


import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, d_model, num_heads, ff_dim,  frozen_model):
        super(TemporalPromptGeneratorModel, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, ff_dim)
      
        self.frozen_model = frozen_model  # 添加冻结的模型

    def call(self, inputs, training):
        prompts = inputs
    
        # 将 prompts 传递给 Transformer 编码器模块
        encoder_output = self.encoder(prompts, training=training)

        batch_size = tf.shape(encoder_output)[0]

# 使用 tf.reshape 将三维张量 reshape 为 (None, 1280) 的二维张量
        encoder_output = tf.reshape(encoder_output, (batch_size, -1))
       
        # 将 Transformer 编码器的输出传递给冻结的模型
        #frozen_model = self.frozen_model(encoder_output.shape[1])
        frozen_output = self.frozen_model(encoder_output )

        return frozen_output

# 参数设置



train_cite_X = np.load(feature_path+'train_cite_X.npy')
test_cite_X = np.load(feature_path+'test_cite_X.npy')

train_cite_y = np.load(feature_path+'day4_train_cite_targets.npy')  
cite_prompt = np.load(feature_path+'cite_prompt.npy')



# 编译模型
num_samples = 1000
num_prompts = 10
prompt_dim = 128
output_dim = 140

frozen_model = keras.Sequential([
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(140, activation='softmax')
])

x_train = np.random.rand(num_samples, num_prompts, prompt_dim)
y_train = np.random.rand(num_samples, output_dim)

d_model = 128 # 模型维度
num_heads = 4   # 注意力头的数量
ff_dim = 32     # Feedforward层的维度



frozen_model.load_weights("/home/dujunjia/bio/solution/cite_cos_model_1.h5")
model = TemporalPromptGeneratorModel(d_model, num_heads, ff_dim,frozen_model)

model.compile(optimizer="adam", loss="mse")
history = model.fit(x_train,y_train,
                    batch_size=64,
                    epochs=1)



model.summary()