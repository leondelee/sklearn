#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:31:16 2018

@author: llw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
digit_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digit_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
x_train=digit_train[np.arange(64)]
y_train=digit_train[64]
x_test=digit_test[np.arange(64)]
y_test=digit_test[64]

from sklearn.cluster import KMeans
km=KMeans(n_clusters=10)
km.fit(x_train)
y_pre=km.predict(x_test)
'''
''' #draw different figures of samples with different clusters
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
plt.subplot(3,2,1)
x1=np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2=np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
tem=zip(x1,x2)
x=np.empty([len(x1),2])
for idx,value in enumerate(tem):
    x[idx,:]=np.array(value)
print(x)
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)
plt.figure()
colors=['b','g','r','c','m','y','k','b']
markers=['o','s','d','v','^','p','*','+']
clusters=[2,3,4,5,8]
subplot_counter=1
sc_scores=[]
for t in clusters:
    subplot_counter+=1
    plt.subplot(3,2,subplot_counter)
    km=KMeans(n_clusters=t).fit(x)
    for idx,label in enumerate(km.labels_):
        plt.plot(x1[idx],x2[idx],color=colors[label],marker=markers[label],ls='None')#linestyle
    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_score=silhouette_score(x,km.labels_,metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s,sigouette coefficient=%0.03f'%(t,sc_score))
    plt.figure()
plt.plot(clusters,sc_scores,'*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()
'''

 #elbow method for deciding number of clusters
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(5.5,6.5,(2,10))
cluster3=np.random.uniform(3.0,4.0,(2,10))
x=np.hstack((cluster1,cluster2,cluster3)).T
plt.scatter(x[:,0],x[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
mendistotions=[]
for k in range(1,10):
    km=KMeans(n_clusters=k)
    km.fit(x)
    mendistotions.append(sum(np.min(cdist(x,km.cluster_centers_,'euclidean'),axis=1))/x.shape[0])
plt.plot(range(1,10),mendistotions,'bx-')
plt.xlabel('k')
plt.ylabel('average dispersion')
plt.title('selecting k with the elbow method')
plt.show()
    
    
