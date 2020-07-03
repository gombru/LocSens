import torch.nn as nn
import MyResNet

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=300)

    def forward(self, image):
        x = self.cnn(image)
        return x