#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:00:27 2020
@author: cindy
"""

import numpy as np
import matplotlib.pyplot as plt


# Linear Separable dataset 
# x1(threshold), x2, x3 input (training data) 
# expected output
dataset = np.array([
    ((1, -0.4, 0.3), -1),
    ((1, -0.3, -0.1), -1),
    ((1, -0.2, 0.4), -1),
    ((1, -0.1, 0.1), -1),
    ((1, 0.9, -0.5), 1),
    ((1, 0.7, -0.9), 1),
    ((1, 0.8, 0.2), 1),
    ((1, 0.4, -0.6), 1)])

'''
# Build another dataset
dataset2 = np.array([
    ((1, 0, 0), -1),
    ((1, 0, 1), 1),
    ((1, 1, 0), 1),
    ((1, 1, 1), 1),
    ((0, 0, 1), -1),
    ((0, 1, 0), -1),
    ((0, 1, 1), 1),
    ((0, 0, 0), -1)])

# Another method
x = np.mat([[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,0,1],[0,1,0],[0,1,1],[0,0,0]])
y = np.mat([[-1],[1],[1],[1],[-1],[-1],[1],[-1],])
'''
# Error Rate
def check_error(w, dataset):
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        #print(x)  --> threshold and training data  
        #print(s)  --> expected output
        # np.sign --> sign function 
        if int(np.sign(w.T.dot(x)))!=s:
            result = x,s
            error+=1
    print("Error rate=%s/%s" % (error, len(dataset)))
    return result
    
# Perceptron
def pla(dataset):
    w = np.zeros(3) # value set by user in the beginning
    # another way to set the value
    # w = np.mat([0.3, 0.3, 0.3])
    while check_error(w, dataset) is not None:
        x, s = check_error(w, dataset)
        w += s * x
    return w
    
w = pla(dataset)

ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter([v[1] for v in ps[:4]], [v[2] for v in ps[:4]], s=10, c='b', marker="o", label='O')
ax1.scatter([v[1] for v in ps[4:]], [v[2] for v in ps[4:]], s=10, c='r', marker="x", label='X')
l = np.linspace(-2,2)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()