#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:31:08 2020
用前九个小时pm2.5来预测第10小时的pm2.5
@author: cindy
"""
import csv
import numpy as np
import pandas as pd

data = pd.read_csv("train.csv", encoding='big5')
# use iloc for slicing rows and columns
# do not slice row, slice 0-2 attributes in pm2_5
pm2_5 = data[data['測項']=='PM2.5'].iloc[:,3:]
#pm2_5.to_csv("train1.csv", sep=',')
#print(pm2_5)

# training data processing
# first 9 hrs as feature
# the 10th hours as lable

tempxlist = [] # training data
tempylist = [] 

for i in range(0,15):
    tempx = pm2_5.iloc[:, i:i+9] # first 9 hrs
    tempx.columns = np.array(range(9))
    tempy = pm2_5.iloc[:, i+9]
    tempy.columns = ['1']
    tempxlist.append(tempx)
    tempylist.append(tempy)
    
x_data = pd.concat(tempxlist)
x = np.array(x_data, float)
y_data = pd.concat(tempylist)
y = np.array(y_data,float)
print(x.shape)
print(y.shape)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1) 
print(x)

# 初始化一个参数矩阵
w=np.zeros((len(x[0])))

#初始化一个learning rate
lr=10
iteration=10000   #迭代10000次
s_grad=np.zeros(len(x[0]))
for i in range(iteration):
    tem=np.dot(x,w)     #&y^*&(预测值)
    loss=y-tem     
    grad=np.dot(x.transpose(),loss)*(-2)
    s_grad+=grad**2
    ada=np.sqrt(s_grad)
    w=w-lr*grad/ada
print(w)
