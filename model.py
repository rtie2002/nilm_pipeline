import torch
from torch import nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, window_length: int):
        super(CNN, self).__init__()
        self.n = 32
        self.window_length = window_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.n, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(self.n, self.n, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.n * self.window_length, 1024)
        self.fc2 = nn.Linear(1024, self.window_length)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device in model.py")
    window_length = 576
    model = CNN(window_length=window_length).to(device)
    summary(model, input_size=(window_length,))
  
   
   
