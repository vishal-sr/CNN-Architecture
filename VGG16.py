# Imports
import torch
import torch.nn as nn

# Model building
class VGG16(nn.Module):
  def __init__(self, img_channels = 3, num_classes = 1000, config = None):
    super(VGG16, self).__init__()
    self.img_channels = img_channels
    self.conv_layers = self.create_conv_layers(config)
    self.fc_layers = nn.Sequential(
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc_layers(x)
    return x

  def create_conv_layers(self, config):
    in_channels = self.img_channels
    layers = []

    for x in config:
      if type(x) == int:
        out_channels = x
        layers += nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                      kernel_size = (3, 3), stride = (1, 1), padding = 'same'),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),
        )
        in_channels = x
      elif x == 'Pool':
        layers += nn.Sequential(
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
        )
    return nn.Sequential(*layers)

# Configurations
# conv. layer - (kernel_size = (3, 3), stride = (1, 1), padding = 'same')
# maxpool - (kernel_size = (2, 2), stride = (2, 2))
# 
# (conv, 64)
# (conv, 64)
# (maxpool)
# (conv, 128)
# (conv, 128)
# (maxpool)
# (conv, 256)
# (conv, 256)
# (conv, 256)
# (maxpool)
# (conv, 512)
# (conv, 512)
# (conv, 512)
# (maxpool)
# (conv, 512)
# (conv, 512)
# (conv, 512)
# (maxpool)
# (fc, 4096)
# (fc, 4096)
# (fc, 1000)


# Configuration list
conv_config = [
               64, 64, 'Pool', 
               128, 128, 'Pool',
               256, 256, 256, 'Pool', 
               512, 512, 512, 'Pool', 
               512, 512, 512, 'Pool',
]


# Model initialization
vgg = VGG16(config = conv_config)

# Model checking
x = torch.rand((32, 3, 224, 224)) # 32 images of 224x224 with 3 channels
print(vgg(x).shape)