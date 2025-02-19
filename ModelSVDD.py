import torch.nn as nn
import torch.nn.functional as F

""" SVDD Model that follow the paper implementations. """

class SVDD(nn.Module):
    def __init__(self, latent_dim, num_filters=2):
        super(SVDD, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=2, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=self.num_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=2, stride=1,
                               padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=self.num_filters)
        self.conv3 = nn.Conv2d(in_channels=self.num_filters, out_channels=1, kernel_size=2, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(1 * 2 * 2, self.latent_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 1 * 2 * 2)
        x = self.fc1(x)
        return x


