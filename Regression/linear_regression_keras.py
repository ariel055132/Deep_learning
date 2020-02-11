#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:06:59 2020
Linear Regression with Keras
@author: cindy
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# y = ax+b, a=slope,b=intercept
rng = np.random.RandomState(1)
x = 10 * rng.rand(200)
y = 2 * x - 5 + rng.randn(200)

#Plot the graph
#plt.scatter(x, y)
#plt.show()

# training data, the first 160 data
x_train, y_train = x[:160], y[:160]
# testing data, the last 40 data
x_test, y_test = x[160:], y[160:]

# Build the model
network = models.Sequential()
network.add(layers.Dense(20, activation='relu', input_dim=1))
network.add(layers.Dense(20, activation='relu'))
# No activation function --> linear layer
network.add(layers.Dense(1))

# choose loss function and optimizing method
# compile have metrics --> results have loss values and accuracy
# otherwise only loss values
network.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
network.fit(x_train, y_train, epochs=20, batch_size=3)
results = network.evaluate(x_test, y_test)
y_pred = network.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred)
plt.show()