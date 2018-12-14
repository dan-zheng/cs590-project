# Train dense neural network on schedule features.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

x_train = np.loadtxt('x_train.out')
y_train = np.loadtxt('y_train.out')
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

print(x_train)
print(y_train)

in_size = x_train.shape[1]
hidden_size = 200
out_size = y_train.shape[0]

model = torch.nn.Sequential(
    torch.nn.Linear(in_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, out_size),
)

opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for _ in range(100):
    opt.zero_grad()
    loss = criterion(model(x_train), y_train)
    loss.backward()
    print(loss)
    opt.step()

