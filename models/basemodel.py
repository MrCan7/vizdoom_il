import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

from models.convLSTM import ConvLSTM
#from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, hidden_dim = 64, action_size = 16):
        super().__init__()

        self.feature_extractor = models.efficientnet_b0(weights='IMAGENET1K_V1') #initalized with imagenet1k weights on default
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[0][0: 6]) #keep the first 6 layers

        self.conv_lstm = ConvLSTM(input_dim = 112, hidden_dim= hidden_dim, kernel_size= (3,3), num_layers=1, batch_first= True, bias=True, return_all_layers= False )
        self.fc = nn.Linear(hidden_dim, action_size)

        self.fc_out_wasd = nn.Linear(hidden_dim* 14*14, 4) #output for wasd 
        self.out_mouse_LR = nn.Linear(hidden_dim* 14*14 , 1 )#output for mouse Left Right Delta
        self.out_mouse_UD = nn.Linear(hidden_dim* 14*14 , 1 )#output for mouse
        self.shoot = nn.Linear(hidden_dim* 14*14 , 1 )#output for shooting
        self.act = torch.nn.Sigmoid()
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)

        x = self.feature_extractor(x)
        _, c, h,w = x.shape
        x= x.view(b,t,c,h,w)
        layer_outputs , last_states = self.conv_lstm(x)
        #take the output of every layer
        lstm_out = layer_outputs[0]#[:, -1, :, :, :]
        lstm_out = torch.flatten(lstm_out, 2)
        
        wasd_out = self.act(self.fc_out_wasd(lstm_out))
        mouse_LR_out = self.out_mouse_LR(lstm_out) 
        mouse_UD_out = self.out_mouse_UD(lstm_out)
        mouse_shoot_out = self.act(self.shoot(lstm_out))

        return wasd_out,mouse_LR_out, mouse_UD_out, mouse_shoot_out      
if __name__ == "__main__":
    x = torch.rand((1,2, 3, 224, 224))
    #print(x.shape)
    model = BaseModel()
    model(x)

    
