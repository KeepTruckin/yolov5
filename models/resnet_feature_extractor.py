import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


RESNET_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101']

return_nodes = {
    'maxpool': 'maxpool',
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
}

class ResNetFeatureExtractor(nn.Module):
    # resnet feature  extractor
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super(ResNetFeatureExtractor, self).__init__()

        if backbone_name not in RESNET_BACKBONES:
            raise ValueError("Invalid backbone name. Must be one of {}".format(RESNET_BACKBONES))

        model = torchvision.models.__dict__[backbone_name](weights='IMAGENET1K_V1' if pretrained else None)

        all_layers = list(model.children())[:-2]
        #
        self.features1 = nn.Sequential(*all_layers[:4])
        self.features2 = all_layers[4]
        self.features3 = all_layers[5]
        self.features4 = all_layers[6]
        self.features5 = all_layers[7]

        # self.features = create_feature_extractor(model, return_nodes=return_nodes)
        self.channel = [i.size(1) for i in self.forward(torch.randn(2, 3, 640, 640))]
        # self.channel = [64, 256, 512, 1024, 2048]


    def forward(self, x):
        x1 =  self.features1(x)
        x2 =  self.features2(x1)
        x3 =  self.features3(x2)
        x4 =  self.features4(x3)
        x5 =  self.features5(x4)
        return [x1, x2, x3, x4, x5]


if __name__ == "__main__":
    model = ResNetFeatureExtractor(backbone_name="resnet50")
    out = model(torch.rand(1, 3, 224, 224))