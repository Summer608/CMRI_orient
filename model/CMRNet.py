from torch import nn


class CMRNet(nn.Module):
    def __init__(self):
        super(CMRNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.fl = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.avgp(x)
        x = x.view(-1, 64*16*16)
        x = self.fc2(self.sigmoid(self.fc1(x)))
        return x

