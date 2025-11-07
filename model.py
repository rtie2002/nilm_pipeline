import torch
from torch import nn
from torch import summary

class CNN(nn.Module):
    def __init__(self,
                input_size):
        super(Proxy_Model, self).__init__()
        self.n = 32
        self.conv1 = nn.Conv1d(1, self.n, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n * 576, 1024)
        self.fc2 = nn.Linear(1024, input_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, self.n * 576)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x