import torch.nn as nn
import MyResNet
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=300)
        self.output_layer = nn.Linear(300, 100000)

    def forward(self, image):
        x = self.cnn(image)
        x = self.output_layer(x)
        return x

class Model_Test(nn.Module):

    def __init__(self):
        super(Model_Test, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=300)

    def forward(self, image):
        x = self.cnn(image)
        return x

class Model_SoftMax(nn.Module):

    def __init__(self):
        super(Model_SoftMax, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=300)
        self.output_layer = nn.Linear(300, 100000)

    def forward(self, image):
        x = self.cnn(image)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x
