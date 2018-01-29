#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:24:30 2018

@author: llw
"""
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
column_names=['scn','ct','ucsz','ucsp','ma','secs','bn','bc','nn','mit','class']
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
#print(data)
x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
#print(x_train,y_train)
#print(y_train.value_counts())
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

lr=LogisticRegression()
sgdc=SGDClassifier()
lr.fit(x_train,y_train)
lr_y_predict=lr.predict(x_test)
sgdc.fit(x_train,y_train)
sgdc_y_predict=sgdc.predict(x_test)

print('accuracy of LR',lr.score(x_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['benign','malignant']))
print('accuracy of SGD',sgdc.score(x_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names=['benign','malignant']))

