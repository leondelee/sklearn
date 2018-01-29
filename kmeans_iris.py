#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:19:16 2018

@author: llw
"""
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
y_predict=knc.predict(x_test)

from sklearn.metrics import classification_report
print('accuracy of knc is',knc.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=iris.target_names))