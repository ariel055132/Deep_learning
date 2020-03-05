#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:03:49 2020
Salary prediction with the data
@author: cindy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Salary_Data.csv')
exp = df['YearsExperience']
salary = df['Salary']

# Plot the graph and output the data to check the characteristics of the data
print(exp)
print("--------------------")
print(salary)
#plt.scatter(salary, exp)
#plt.xlabel("Salary")
#plt.ylabel("Experience")
#plt.show()

# Ordinary least square
def linear_regression(x,y):
  x=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y=y[:,np.newaxis]
  beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return beta

by_hand = linear_regression(salary, exp)
# y-intercept
print("y-intercept:", by_hand[0])
# slope
print("slope:", by_hand[1])

# plot the result  
xs = np.linspace(38000, 130000, 3000)
ys = by_hand[0] + by_hand[1] * xs
plt.scatter(salary, exp)
plt.plot(xs, ys, 'r' , linewidth=3)
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()