import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# change the weight with method gradient descent
weight = 1.0  # random value, weight will change to suitable value in calculation


# loss = (yHat-y)^2 = (x*w-y)^2
def calculateYHat(x):
    return x * weight


def calculateLoss(x, y):
    yHat = calculateYHat(x)
    return (yHat - y) * (yHat - y)


# partial derivative
def calculateGradient(x, y):
    return 2 * x * (x * weight - y)



print("predict （before training", 4, calculateYHat(4))
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        gradient = calculateGradient(x_val, y_val)
        weight = weight - 0.01 * gradient  # minus because it needs to walk down
        print("\tgrad: ", x_val, y_val, gradient)
        loss = calculateLoss(x_val, y_val)
    print("Epoch: ", epoch, "weight: ", weight, "loss: ", loss)

print("predict (after training)",  4, calculateYHat(4))
