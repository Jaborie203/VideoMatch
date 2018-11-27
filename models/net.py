from torchvision.models import resnet101
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = resnet101(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
if __name__ == '__main__':
    net = Net()
    x = torch.randn(13,3,854,480)
    y = net(x)
    print(y.shape)
