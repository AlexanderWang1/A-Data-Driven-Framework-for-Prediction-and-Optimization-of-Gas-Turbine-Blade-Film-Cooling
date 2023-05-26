import numpy as np
import re
import tensorflow as tf
import numpy as np
import keras

from keras.optimizers import adam_v2


from keras.models import Sequential,Input,Model
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau

import time
import json
import urllib3
import base64
import pandas as pd
import numpy as np
import warnings
import os
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
import matplotlib as mpl
import scipy.special
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import KFold,train_test_split,GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
import pickle
import seaborn as sns
import random
import scipy.io as sio

import keras
from keras.layers import LSTM,GRU
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D,LayerNormalization
from keras.layers import Dense, Activation, Convolution1D,Convolution2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation,Input
from keras.layers import Conv2DTranspose,Conv2D

from keras.callbacks import LearningRateScheduler
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D


import numpy as np
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import keras.backend as K

from keras.callbacks import LearningRateScheduler

label_train=np.load('label_train.npy')
label_test=np.load('label_test.npy')
y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')
X_train=np.load('x_train.npy')
X_test=np.load('x_test.npy')



def scheduler(epoch):

    # 每隔100个epoch，学习率减小为原来的1/10

    if epoch % 1000 == 0 and epoch != 0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr * 0.1)

        print("lr changed to {}".format(lr * 0.1))

    return K.get_value(model.optimizer.lr)

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr




            


print(np.shape(X_train))
print(np.shape(y_train))
print(np.shape(label_train))

# optimizer = adam_v2.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

inputB = Input(shape=(256,3,))


# Adam1 = adam_v2.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

inputB = Input(shape=(256,3,))

drop=0.1

z = GRU(128, input_dim=3, input_length=256, return_sequences=True, dropout=0.1)(inputB)
z = Dropout(drop)(z)
z = GRU(256, return_sequences=True, dropout=0.1)(z)


generator = Model(inputs=[inputB], outputs=z)
generator.summary()
# reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')

drop = 0.1
lr = 0.01
# Adam1 = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
img = Input(shape=(256,256,1)) # 输入 （28，28，1）

label = Input(shape=(17,), dtype='int32')

label_embedding = Dense(65536, activation="relu") (Flatten()(label))


flat_img = Flatten()(img)


print(flat_img.shape)
print(label_embedding.shape)

model_input = multiply([flat_img, label_embedding])


# inputA = Input(shape=16384,)
x = tf.reshape(model_input, [-1,256,256,1])
print(z.shape)
# inputB = Input(shape=(2,))


# x = Conv2D(256,(3,3),strides=(2, 2),padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x7 = Dropout(drop)(x)

x = Conv2D(128,(3,3),strides=(2, 2),padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x6 = Dropout(drop)(x)

x = Conv2D(64,(3,3),strides=(2, 2),padding='same')(x6)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x5 = Dropout(drop)(x)

x = Conv2D(32,(3,3),strides=(2, 2),padding='same')(x5)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x4 = Dropout(drop)(x)

x = Conv2D(16,(3,3),strides=(2, 2),padding='same')(x4)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x3 = Dropout(drop)(x)

x = Conv2D(8,(3,3),strides=(2, 2),padding='same')(x3)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x2 = Dropout(drop)(x)


x = Conv2D(4,(3,3),strides=(2, 2),padding='same')(x2)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x2 = Dropout(drop)(x)

x = Conv2D(2,(3,3),strides=(2, 2),padding='same')(x2)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x2 = Dropout(drop)(x)

x = Conv2D(1,(3,3),strides=(2, 2),padding='same')(x2)
# x = MaxPooling2D(pool_size = (2 ,2))(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(drop)(x)


validity = Flatten()(x)

discriminator = Model([img, label], validity)
# discrimator = Model(inputs=[inputA], outputs=x)
discriminator.summary()


optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# discriminator
def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
lr_metric = get_lr_metric(optimizer)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=[lr_metric, 'accuracy'])

generator.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=[lr_metric])



batch_size = 8
# sample_interval = 200

# Load the dataset

label_train=np.load('label_train.npy')
label_test=np.load('label_test.npy')
y_train=np.load('y_train.npy')
y_test=np.load('y_test.npy')
X_train=np.load('x_train.npy')
X_test=np.load('x_test.npy')



print(np.shape(X_train))
print(np.shape(y_train))
    
    


batch_count = X_train.shape[0]//batch_size

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(4000):

    if epoch % 1000 == 0 and epoch != 0:
        lr = K.get_value(discriminator.optimizer.lr)
        lr = lr * 0.1 
        K.set_value(discriminator.optimizer.lr, lr) 


    # ---------------------
    #  Train Discriminator
    # ---------------------
    # Select a random batch of images
    for i in range(batch_count):
        idx = np.random.randint(0, X_train.shape[0], batch_size)  
        # imgs = X_train[idx]
        imgs, labels, labels_flatten = y_train[idx], X_train[idx], label_train[idx]
        # Generate a batch of new images
        gen_imgs = generator.predict(labels)
        # Train the discriminator

        d_loss_real = discriminator.train_on_batch([imgs, labels_flatten], valid)  
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels_flatten], fake)  
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Generator
        samidx = np.random.randint(0, X_train.shape[0], batch_size) 
        samimgs, samlabels = y_train[samidx], X_train[samidx]
        g_loss = generator.train_on_batch(samlabels, samimgs)
    # Plot the progress
    if epoch % 1 == 0:
        print("%d [D loss: %f, lr: %f, acc.: %.2f%%] [G loss: %f, lr: %f]" % (
        epoch, d_loss[0], d_loss[1], 100 * d_loss[2], g_loss[0], g_loss[1]))
    # If at save interval => save generated image samples
#     if epoch % sample_interval == 0:
#         sample_images(epoch)
generator.optimizer = None
generator.compiled_loss = None
generator.compiled_metrics = None

generator.save('cgan_gee3_2D_hole11_modelcgan_generator.h5')
discriminator.save('cgan_gee3_2D_hole11_modelcgan_discriminator.h5')

generator.compile(loss='mae')
preds0 = generator.evaluate(X_train, y_train, batch_size = 8)
print(preds0)
preds = generator.evaluate(X_test, y_test, batch_size =8)
print(preds)
Y_pred0 = generator.predict(X_train)
Y_pred = generator.predict(X_test)
print(np.shape(X_test))
print(np.shape(Y_pred))

