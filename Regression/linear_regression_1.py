#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:47:53 2020
Linear Regression Implementation
@author: cindy
"""
import numpy as np
import matplotlib.pyplot as plt

# Ordinary Least Square Formula
def linear_regression(x,y):
    # concatenate: join a sequence of arrays along an existing axis
    x = np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
    # newaxis: add the dimension of the matrix
    y = y[:,np.newaxis] 
    # matmul: matrix multiplication
    # linalg: Linear algebra
    # inv: inverse of the matrix (A^-1)
    # matrix.T: return the transpose of the matrix
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y) 
    return beta
    
    
# y = ax + b, a = slope, b = intercept
rng = np.random.RandomState(1) # Random seed
# Create some points
x = 10 * rng.rand(50) # generate data between [0,1) dimension
y = 2*x -5 + rng.randn(50) # a set of data with a standard normal distribution
'''
#Plot the graph and check whether the points can be generated or not
plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
'''
# sub x, y into function linear_regression
by_hand = linear_regression(x,y)
# y-intercept
print(by_hand[0])
# slope
print(by_hand[1])

# linspace: return evenly spaced number over a specified interval
xs = np.linspace(0,10,200) # 200 numbers
# Follow the formula: y = ax + b 
ys = by_hand[1]*xs + by_hand[0] 
# Plot the graph
# alpha: transparency of the nodes
plt.scatter(x,y,alpha=0.3) 
plt.plot(xs,ys,'r')
plt.show()
