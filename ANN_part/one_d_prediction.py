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

X_train=np.load('label_train.npy')
X_test=np.load('label_test.npy')
y_train=np.load('y_train_d11.npy')
y_test=np.load('y_test_d11.npy')


    
drop = 0.5
lr = 0.1
Adam1 = adam_v2.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

inputB = Input(shape=(17,))



z = Dense(256, activation="relu")(inputB)
z = BatchNormalization()(z)
z = Dropout(drop)(z)








z = Dense(11, activation="sigmoid")(z)



def scheduler(epoch):


    if epoch % 1000 == 0 and epoch != 0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr * 0.1)

        print("lr changed to {}".format(lr * 0.1))

    return K.get_value(model.optimizer.lr)



def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


lr_metric = get_lr_metric(Adam1)

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
reduce_lr = LearningRateScheduler(scheduler)
model = Model(inputs=[inputB], outputs=z)
model.compile(loss='mse', optimizer=Adam1,metrics = [lr_metric])
model.summary()

model.fit(X_train, y_train,batch_size = 8, epochs = 4000,  callbacks=[reduce_lr])
model.save('gee3_hole11_position_model.h5')

model=keras.models.load_model('gee3_hole11_position_model.h5', compile=False)


model.compile(loss='mae')
print(np.shape(X_train))
preds0 = model.evaluate(X_train, y_train, batch_size = 8)
print(preds0)
preds = model.evaluate(X_test, y_test, batch_size =8)
print(preds)
Y_pred0 = model.predict(X_train)
Y_pred = model.predict(X_test)
print(np.shape(X_test))
print(np.shape(Y_pred))

