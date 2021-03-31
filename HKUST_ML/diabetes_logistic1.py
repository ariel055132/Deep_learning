from torch import nn, optim, from_numpy
import numpy as np

xy_data = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy_data[:, 0:-1])
y_data = from_numpy(xy_data[:, [-1]])

# add one more Linear layer to the model
# Check whether the loss is decreased or not
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(8, 6)
        self.layer2 = nn.Linear(6, 4)
        self.layer3 = nn.Linear(4, 2)
        self.layer4 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output1 = self.sigmoid(self.layer1(x))
        output2 = self.sigmoid(self.layer2(output1))
        output3 = self.sigmoid(self.layer3(output2))
        yHat = self.sigmoid(self.layer4(output3))
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