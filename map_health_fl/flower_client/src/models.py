# from torch import nn
# # import torch.nn.functional as F


# class Net(nn.Module):
#     """
#     A simple CNN adapted for MedMNIST.
#     The final layer will be dynamically adjusted based on the global number of classes.
#     """
#     def __init__(self, in_channels=3, num_classes=21):
#         super(Net, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU())

#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.layer3 = nn.Sequential(
#             nn.Conv2d(16, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
        
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())

#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, num_classes))

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

import torchvision.models as models
from torch import nn


class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
