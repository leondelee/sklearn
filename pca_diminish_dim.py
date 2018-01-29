#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:58:12 2018

@author: llw
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
digit_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digit_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
digit_train.to_csv('digit_train.csv')
digit_test.to_csv('digit_test.csv')
'''
digit_train=pd.read_csv('digit_train.csv',header=None)
digit_test=pd.read_csv('digit_test.csv',header=None)
x_train=digit_train[np.arange(1,65)][1:digit_train.shape[0]]
y_train=digit_train[65][1:digit_train.shape[0]]

from sklearn.decomposition import PCA
esti=PCA(n_components=2)
x_pca=esti.fit_transform(x_train)


colors= ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
for i in range(len(colors)):
    px=x_pca[:,0][y_train.as_matrix()==i]
    py=x_pca[:,1][y_train.as_matrix()==i]
    plt.scatter(px,py,c=colors[i])
plt.legend(np.arange(0,10).astype(str))
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.show()