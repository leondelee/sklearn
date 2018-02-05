#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:46:37 2018

@author: llw
"""
import pandas as pd
import numpy as np
digit_train=pd.read_csv('digit_train.csv',header=None)
digit_test=pd.read_csv('digit_test.csv',header=None)
x_train=digit_train[np.arange(1,digit_train.shape[1]-1)][1:digit_train.shape[0]]
y_train=digit_train[digit_train.shape[1]-1][1:digit_train.shape[0]]
x_test=digit_test[np.arange(1,digit_train.shape[1]-1)][1:digit_train.shape[0]]
y_test=digit_test[digit_train.shape[1]-1][1:digit_train.shape[0]]
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
clf=Pipeline([('svc',SVC())])
paras={'svc__gamma':np.array([0.001,0.01,0.1,1]),'svc__kernel':['rbf','linear','poly'],'svc__C':np.array([1,2,3,4])}
from sklearn.grid_search import GridSearchCV
gs=GridSearchCV(clf,paras,n_jobs=-1,cv=5,verbose=2,refit=True)
gs.fit(x_train,y_train)
print(gs.best_score_,gs.best_params_)
print(gs.score(x_test,y_test))