#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:22:43 2018

@author: llw
"""
import numpy as np
def add_narray(a,b,axis):
    if axis==0:
        return np.hstack((a,b))
    elif axis==1:
        return np.vstack((a,b))
    else:
        print('axis must be 0 or 1')
xtrain=np.array([6,8,10,14,18]).reshape(-1,1)
#xtrain=add_narray(np.ones([xtrain.shape[0],1]),xtrain,0)
ytrain=np.array([7,9,13,17.5,18]).reshape(-1,1)
xtest=np.array([6,8,11,16]).reshape(-1,1)
#xtest=add_narray(np.ones([xtest.shape[0],1]),xtest,0)
''' # linear
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)
#print('coefficients are',lr.coef_)
#print('prediction is',lr.predict(xtest))
'''
#polynomial
from sklearn.preprocessing import PolynomialFeatures
poly2=PolynomialFeatures(degree=4)
xtrain_2=poly2.fit_transform(xtrain)
xtest_2=poly2.transform(xtest)
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr2=LinearRegression()
lr1.fit(xtrain,ytrain)
lr2.fit(xtrain_2,ytrain)

import matplotlib.pyplot as plt
plt.scatter(xtrain,ytrain)
xx=np.linspace(0,25,100)
xx1=xx.reshape(-1,1)
xx2=poly2.transform(xx1)
yy1=lr1.predict(xx1)
yy2=lr2.predict(xx2)
plt1,=plt.plot(xx,yy1,label='degree 1')  #generate line2D object
plt2,=plt.plot(xx,yy2,label='degree 2')
plt.xlabel('Diameter of pizza')
plt.ylabel('Price of pizza')
plt.legend(handles=[plt1,plt2])
plt.show()

