# Implement linear regression with scikit learn
# Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variable.
# Finding the curve that best fits your data is called regression, and when that curve is a straight line, it's called linear regression.
# 在資料點中找出規律、畫出一條直線

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from pylab import mpl
import time

# formula:   y_data = b + w * x_data (relationship between x and y)
# we need to find the value of b and w from y_data and x_data
# Gradient Descent will be used 

# data 
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.] # 10 x_data
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.] # 10 y_data
# turn the list into array
x_d = np.asarray(x_data) 
y_d = np.asarray(y_data) 

x = np.arange(-200, -100, 1) # bias
y = np.arange(-5, 5, 0.1) # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)

# Loss function
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0 # meshgrid result: y is row, x is col
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] /= len(x_data)

# partial differentiate b and w
# Linear Regression
b = -2 # initial b
w = 0.01   # initial w
lr = 0.000005 # learning rate
iteration = 1400000 # epochs

b_hist = [b]
w_hist = [w]
loss_hist = []
start = time.time()

# differentiate two variables and obtain their gradient
for i in range(iteration):
    m = float(len(x_d))
    y_hat = w * x_d  +b
    loss = np.dot(y_d - y_hat, y_d - y_hat) / m
    grad_b = -2.0 * np.sum(y_d - y_hat) / m
    grad_w = -2.0 * np.dot(y_d - y_hat, x_d) / m

    # update parameters
    b -= lr * grad_b
    w -= lr * grad_w

    b_hist.append(b)  # insert
    w_hist.append(w)  # insert
    loss_hist.append(loss) # insert

    # Show some results after 10000 epochs
    if i % 10000 == 0:
        print("Step %i, w: %0.4f, b: %.4f, Loss: %.4f" % (i, w, b, loss))

end = time.time()
print("Time needed: ", end-start)

# Plot the figure
f1 = plt.figure(1)
plt.subplot(1,2,1)
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, mew=3, color='orange')
plt.plot(b_hist, w_hist, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel('b')
plt.ylabel('w')
plt.title('Linear Regression')

plt.subplot(1,2,2)
loss = np.asarray(loss_hist[2:iteration]) # start from 2 as 1 is defined by us
plt.plot(np.arange(2,iteration), loss)
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# The result is bad. Cannot reach the optimum result.
print("-------------------------------------------------------")

# Renew the model
f2 = plt.figure(2)
# Linear regression
b = -120
w = -4
lr = 1
iteration = 100000
b_hist = [b]
w_hist = [w]

lr_b = 0
lr_w = 0
start = time.time()
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - n - w * x_data[n])*1.0
        w_grad = w_grad - 2.0 * (y_data[n] - n - w * x_data[n])*x_data[n]
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    # update parameter 
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad

    b_hist.append(b)
    w_hist.append(w)

end = time.time()
print("Time needed: ", end-start)

# Plot the figure
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))  # 填充等高线
plt.plot([-188.4], [2.67], 'x', ms=12, mew=3, color="orange")
plt.plot(b_hist, w_hist, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel('b')
plt.ylabel('w')
plt.title("Linear Regression")

plt.show()