import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# loss = (yHat-y)^2 = (x*w-y)^2
def calculateYHat(x):
    return x*weight

def calculateLoss(x,y):
    yHat = calculateYHat(x)
    return (yHat-y)*(yHat-y)

weight_list = []
mse_list = []

# numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
# Return evenly spaced values within a given interval.
# Values are generated within the half-open interval [start, stop)
# Can run for loop in float
for weight in np.arange(0.0, 4.0, 0.1):
    print("weight= ", weight)
    loss_sum = 0 # for MSE Calculation
    for x_val, y_val in zip(x_data, y_data):
        yHat = calculateYHat(x_val)
        loss = calculateLoss(x_val, y_val)
        loss_sum += loss
        print("\t", x_val, y_val, yHat, loss)
    mse_loss = loss_sum/3
    print("MSE= ", mse_loss)
    weight_list.append(weight)
    mse_list.append(mse_loss)

plt.plot(weight_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('weight')
plt.show()