import torch.nn as nn


class MyMLP(nn.Module):
  def __init__(self):
    super(MyMLP, self).__init__()
    self.fc_layers = nn.Sequential(
                nn.Linear(75, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(32, 5)
            )

  def forward(self, x):
    x = self.fc_layers(x)
    return x


class MyCNN(nn.Module):
  def __init__(self):
    super(MyCNN, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv1d(6, 12, 5)
    self.fc1 = nn.Linear(in_features=12*14, out_features=80)
    self.fc2 = nn.Linear(128, 5)
    self.RELU = nn.ReLU()

  def forward(self, x):
    x = self.pool(self.RELU(self.conv1(x)))
    x = self.pool(self.RELU(self.conv2(x)))
    x = x.view(-1, 12*14)
    x = self.RELU(self.fc1(x))
    x = self.fc2(x)
    return x
