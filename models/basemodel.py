import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = models.efficientnet_b0() #initalized with imagenet1k weights on default

        #keep the first 6 layers
        self.feature_extractor.get_submodule("features")[6] = nn.Identity()
        self.feature_extractor.get_submodule("features")[7] = nn.Identity()
        self.feature_extractor.get_submodule("features")[8] = nn.Identity()

if __name__ == "__main__":

    model = BaseModel()


    
