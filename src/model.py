import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

#default cnn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#simple cnn		
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2400, 50)
        self.fc3 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv2(x)), 2))
        x = x.view(-1, 2400)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
		
#simple3 cnn		
class SimpleNet3(nn.Module):
    def __init__(self):
        super(SimpleNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=5)
        self.conv2 = nn.Conv2d(18, 36, kernel_size=5)
        self.conv3 = nn.Conv2d(36, 40, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.conv2_bn = nn.BatchNorm2d(36)
        self.fc1 = nn.Linear(288, 100)
        self.fc3 = nn.Linear(100, nclasses)
        self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
		
#simple5 cnn		
class SimpleNet5(nn.Module):
    def __init__(self):
        super(SimpleNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=5)
        self.conv2 = nn.Conv2d(18, 36, kernel_size=5)
        self.conv3 = nn.Conv2d(36, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 72, kernel_size=1)
        self.conv5 = nn.Conv2d(72, 128, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.conv4_drop = nn.Dropout2d()
        self.conv5_drop = nn.Dropout2d()
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(72)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152, 100)
        self.fc3 = nn.Linear(100, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

#big cnn		
class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=6)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=6)
        self.conv3 = nn.Conv2d(30, 42, kernel_size=3)
        self.conv4 = nn.Conv2d(42, 64, kernel_size=1)
        self.conv5 = nn.Conv2d(64, 72, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.conv4_drop = nn.Dropout2d()
        self.conv5_drop = nn.Dropout2d()
        self.conv3_bn = nn.BatchNorm2d(42)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(72)
        self.fc1 = nn.Linear(288, 100)
        self.fc3 = nn.Linear(100, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
		
#multi-scale cnn		
class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 18, kernel_size=5)
        self.conv2 = nn.Conv2d(18, 20, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(662, 50)
        self.fc3 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x1 = F.relu(F.max_pool2d(x, 4))
        x2 = F.relu(F.max_pool2d(self.conv3_drop(self.conv2(x)), 2))
        #print(torch.cat([x1.view(x1.shape[0], -1),x2.view(x2.shape[0], -1)], 1).shape)
        x = torch.cat([x1.view(x1.shape[0], -1),x2.view(x2.shape[0], -1)], 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
		
