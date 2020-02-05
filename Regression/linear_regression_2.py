#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:09:56 2020
Implement linear regression with scikit learn
@author: cindy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
#plt.scatter(x,y)
models = LinearRegression(fit_intercept=True)
# Add dimension of the x array to 2d (scikit learn needs data in 2d)
new_x = x[:,np.newaxis] 
model = models.fit(new_x,y)
print(model.coef_[0]) # slope
print(model.intercept_) # y intercept
xfit = np.linspace(0,10,1000)
yfit = model.predict(xfit[:,np.newaxis])
plt.scatter(x,y,alpha=0.3)
plt.plot(xfit,yfit,'r')
