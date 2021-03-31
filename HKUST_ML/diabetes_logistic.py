# from numpy : Creates a Tensor from a numpy.ndarray
from torch import nn, optim, from_numpy
import numpy as np

xy_data = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
print(xy_data.shape)  # 759, 9: 759 rows, 9 columns
print(from_numpy(xy_data[0]))  # first row of data
print(from_numpy(xy_data[:]))  # print all the rows of data
x_data = from_numpy(xy_data[:, 0:-1])
y_data = from_numpy(xy_data[:, [-1]])
#print(x_data)
#print(y_data)
print(x_data.shape)  # 759, 8
print(y_data.shape)  # 759, 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(8, 6) # 8 input and 6 output (6 output will become the input of layer 2)
        self.layer2 = nn.Linear(6, 3) # 6 input and 3 output (same as above)
        self.layer3 = nn.Linear(3, 1) # 3 input and 1 output (same as above)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output1 = self.sigmoid(self.layer1(x))
        output2 = self.sigmoid(self.layer2(output1))
        yHat = self.sigmoid(self.layer3(output2))
        return yHat

model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    yHat = model(x_data)
    loss = criterion(yHat, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
