import torch
from torch import nn
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

weight = Variable(torch.Tensor([1.0]), requires_grad=True)

def calculateYHat(x):
    return x*weight

def calculateLoss(x, y):
    yHat = calculateYHat(x)
    return (yHat-y)*(yHat-y)

print("predict （before training", 4, calculateYHat(4))

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        loss = calculateLoss(x_val,y_val)
        loss.backward()
        print("\tgrad: ", x_val, y_val, weight.grad.data[0])
        weight.data = weight.data - 0.01 * weight.grad.data
        weight.grad.data.zero_()
    print("epoch: ", epoch, loss.data[0])
print("predict (after training)",  4, calculateYHat(4).data[0])