# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# (C - convolutional layer, in_channels, out_channels, kernel_size, Stride, Pooling)
# (P - pooling layer, Stride, Pooling)
# 
# (C, 1, 6, 5, 1, 0) -> 28x28
# (P, 2, 0) -> 14x14
# (C, 6, 16, 5, 1, 0) -> 10x10
# (p, 2, 0) -> 5x5
# (C, 16, 120, 5, 1, 0) -> 1x1
# (Linear layer, 84neurons)
# (Linear layer, 10neurons)


# Model building
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), 
                           stride = (1, 1), padding = (2, 2))
    self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), 
                           stride = (1, 1), padding = (0, 0))
    self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5, 5), 
                           stride = (1, 1), padding = (0, 0))
    self.pool = nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2))
    self.fc1 = nn.Linear(120, 84)
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.tanh(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = F.tanh(x)
    x = self.pool(x)
    x = self.conv3(x)
    x = F.tanh(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    x = F.tanh(x)
    x = self.fc2(x)
    return x

# Model initialization
lenet = LeNet()

# Model checking
x = torch.rand((128, 1, 28, 28)) # 128 images of 28x28 pixels with 1 channel.
print(lenet(x).shape)