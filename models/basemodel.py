import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

from convLSTM import ConvLSTM
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = models.efficientnet_b0(weights='IMAGENET1K_V1') #initalized with imagenet1k weights on default
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[0][0: 6]) #keep the first 6 layers

        #self.conv_lstm = ConvLSTM()

    def forward(self, x):
        x = self.feature_extractor(x)
        return x 
if __name__ == "__main__":
    x = torch.rand((1, 3, 280, 150))
    #b, t, c, h, w = x.shape
    #x = x.view(b*t, c, h, w)
    #print(x.shape)
    model = BaseModel()
    print(model(x).shape)

    
