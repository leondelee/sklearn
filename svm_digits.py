#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 09:04:15 2018

@author: llw
"""
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
digits=load_digits()
print(digits.target_names)

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

lsvc=SVC(kernel='rbf')
lsvc.fit(x_train,y_train)
y_predict=lsvc.predict(x_test)

from sklearn.metrics import classification_report
print('accuracy of svm',lsvc.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))