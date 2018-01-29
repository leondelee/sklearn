#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:45:11 2018

@author: llw
"""
'''
can't load data
'''
import pandas as pd
titanic=pd.read_csv('titanic.txt')
y=titanic['survived']
x=titanic.drop(['row.names','name','survived'],axis=1)
x['age'].fillna(x['age'].mean(),inplace=True)
x.fillna('unknow',inplace=True)
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=33)
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer()
xtrain=vec.fit_transform(xtrain.to_dict(orient='record'))
xtest=vec.transform(xtest.to_dict(orient='record'))
print(xtrain)