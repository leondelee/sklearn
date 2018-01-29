#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:21:51 2018

@author: llw
"""

from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
''' #using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer()
x_train=count_vec.fit_transform(x_train)
x_test=count_vec.transform(x_test)
'''
#using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfi_vec=TfidfVectorizer(analyzer='word',stop_words='english')
x_train=tfi_vec.fit_transform(x_train)
x_test=tfi_vec.transform(x_test)
from sklearn.naive_bayes import MultinomialNB
nbm=MultinomialNB()
nbm.fit(x_train,y_train)
pre=nbm.predict(x_test)
from sklearn.metrics import classification_report
print('the accuracy of nbm is',nbm.score(x_test,y_test))
print(classification_report(y_test,pre,target_names=news.target_names)) 