import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = models.efficientnet_b0() #initalized with imagenet1k weights on default




if __name__ == "__main__":

    model = BaseModel()
    for name, param in model.named_parameters():
        print(name)
