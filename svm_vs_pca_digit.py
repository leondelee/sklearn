#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:46:26 2018

@author: llw
"""
import pandas as pd
import numpy as np
digit_train=pd.read_csv('digit_train.csv',header=None)
digit_test=pd.read_csv('digit_test.csv',header=None)
#print(digit_train.shape)  #3824*66
x_train=digit_train[np.arange(1,digit_train.shape[1]-1)][1:digit_train.shape[0]]
y_train=digit_train[digit_train.shape[1]-1][1:digit_train.shape[0]]
x_test=digit_test[np.arange(1,digit_train.shape[1]-1)][1:digit_train.shape[0]]
y_test=digit_test[digit_train.shape[1]-1][1:digit_train.shape[0]]
#print(y_train)
from sklearn.decomposition import PCA
pca=PCA(n_components=20)
xpc_train=pca.fit_transform(x_train)
xpc_test=pca.transform(x_test)
from sklearn.svm import SVC
clf_all=SVC(kernel='linear')
clf_pc=SVC(kernel='linear')
from sklearn.metrics import classification_report
#all
clf_all.fit(x_train,y_train)
y_all_pre=clf_all.predict(x_test)
print('the accuracy of svm with all the features is',clf_all.score(x_test,y_test))
print(classification_report(y_test,y_all_pre,target_names=np.arange(0,10).astype(str)))

#20 pc
clf_pc.fit(xpc_train,y_train)
y_pc_pre=clf_pc.predict(xpc_test)
print('the accuracy of svm with pincipal components is',clf_pc.score(xpc_test,y_test))
print(classification_report(y_test,y_pc_pre,target_names=np.arange(0,10).astype(str)))