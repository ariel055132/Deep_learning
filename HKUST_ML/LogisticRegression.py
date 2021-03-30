from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim  # package implementing various optimization algorithms

x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1) # One input and one output

    def forward(self, x):
        yHat = sigmoid(self.linear(x))
        return yHat

model = Model()
# Loss function (Binary Cross Enrtopy)
criterion = nn.BCELoss(reduction='mean')
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # forward stage
    # compute predicted y bu passing x to the model
    yHat = model(x_data)

    # Compute and print training loss
    loss = criterion(yHat, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # zero gradients
    # backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')