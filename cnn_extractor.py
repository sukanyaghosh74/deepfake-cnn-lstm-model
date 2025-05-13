import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
            return features.view(features.size(0), -1) 
