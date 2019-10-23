# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:23:42 2019

@author: homestead
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
 
from keras.layers import Dense
from keras.models import Sequential
 
#import
 dataset = pd.read_csv("iris_dataset.csv")
 
 #shuffling
 dataset = dataset.sample(frac=1).reset_index(drop=True)
 #splitting
 x = dataset.iloc[:,0:4]
 y = dataset.iloc[:,4:5]

#scalling
 scaler = MinMaxScaler()
 x = scaler.fit_transform(x)
#one hot encoding
 y = to_categorical(y, num_classes=3)
 
 #traing test spliting
 
 xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2,
                                                 random_state = 40)
 #algo selection
 
model = Sequential()

#layer definitation
model.add(Dense(100, activation= 'relu', input_dim=4))

model.add(Dense(50, activation= 'relu'))

model.add(Dense(30, activation= 'relu'))

model.add(Dense(3, activation= 'softmax')) 

#compilation of model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#input to model
model.fit(xtrain, ytrain, epochs=5, batch_size=10)


model.summary()