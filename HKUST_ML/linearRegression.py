# Linear regression with PyTorch
import torch
from torch import nn
from torch.autograd import Variable
from torch import tensor

# Define data (3*1)
# same as x_data = [1.0, 2.0, 3.0]
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


# Model class in PyTorch way
class Model(nn.Module):
    def __init__(self):
        # Constructor, initialize nn.Linear module
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and One out (input side and output side)

    # Must have
    def forward(self, x):
        yHat = self.linear(x)
        return yHat


model = Model()
# MSELoss
criterion = torch.nn.MSELoss(size_average=False)
# lr = learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass
    yHat = model(x_data)
    # Calculate and print the training loss
    loss = criterion(yHat, y_data)
    print(epoch, loss.item())
    # Zero gradients, backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = tensor([[4.0]])
yHat = model(hour_var)
print(model(hour_var).data[0][0].item())
