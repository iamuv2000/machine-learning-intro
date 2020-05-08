#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:36:54 2020

@author: yuvrajsingh
"""


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[: ,-1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#OneHotEncoding
#Encode country and gender
columnTransformer_x_1 = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder='passthrough')
X = np.array(columnTransformer_x_1.fit_transform(X), dtype = np.str)

#Avoid dummy variable trap
X = X[: , 1:]

#Splitting data into training and test set
from sklearn.model_selection import train_test_split 
X_train , X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#ANN 
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 12))

#Adding the second hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))


#Adding the second hidden layer
classifier.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Making the prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)