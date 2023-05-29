# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('D:\DL Practical\BostonHousingData.csv')
df
x = df.drop("MEDV", axis=1).values
y = df["MEDV"].values
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
def shape():
    print("x_train Shape :",x_train.shape)
    print("x_test Shape :",x_test.shape)
    print("y_train shape :",y_train.shape)
    print("y_test shape :",y_test.shape)
shape()
mean=x_train.mean(axis=0)
std=x_train.std(axis=0)
x_train=(x_train-mean)/std
x_test=(x_test-mean)/std
x_train[0]
y_train[0]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(x_train[0].shape)))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,validation_data=(x_test, y_test))
x_test[8]
test_input=[[-0.42101827, -0.50156705, -1.13081973, -0.25683275, -0.55572682,
0.19758953, 0.20684755, -0.34272202, -0.87422469, -0.84336666,
-0.32505625, 0.41244772, -0.63500406]]
print("Actual Output :",y_test[8])
print("Predicted Output :",model.predict(test_input))
mse_nn,mae_nn=model.evaluate(x_test,y_test)
print('Mean squared error on test data :',mse_nn)
print('Mean absolute error on test data :',mae_nn)
from sklearn.metrics import r2_score
y_dl=model.predict(x_test)
r2=r2_score(y_test,y_dl)
print('R2 Score :',r2)