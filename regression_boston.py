#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:09:09 2018

@author: llw
"""

from sklearn.datasets import load_boston
boston=load_boston()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)
from sklearn.preprocessing import StandardScaler
ssx=StandardScaler()
ssy=StandardScaler()
x_train=ssx.fit_transform(x_train)
x_test=ssx.transform(x_test)
y_train=ssy.fit_transform(y_train.reshape(-1,1))
y_test=ssy.transform(y_test.reshape(-1,1))

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
lr_y_predict=lr.predict(x_test)
from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(x_train,y_train)
sgdr_y_predict=sgdr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('default measurement is',lr.score(x_test,y_test))
print('r2 score is',r2_score(y_test,lr_y_predict))
