from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.out_channels = 2048
    def forward(self, x):
        """
        x: original image, [B, 3, H, W]
        return: feature map, [B, 2048, H/32, W/32]
        """
        return self.body(x)

if __name__ == '__main__':
    import torch
    x = torch.randn(4, 3, 640, 640)
    model = Backbone()
    output = model(x)
    print(output.shape)