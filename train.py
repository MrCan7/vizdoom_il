from models.basemodel import BaseModel
from data.data_prep import get_train_val_test
from data.data_loader import VizDoomData
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import random 
import wandb 
import vizdoom as vzd

REWARD = -10
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def play_one_episode(model, device, epoch):
    model.eval()
    game = vzd.DoomGame()
    game.load_config("/home/stannis/puren/vizdoom_il/scenarios/deathmatch.cfg")
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_render_all_frames(True)
    #game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    transform = transforms.CenterCrop((224, 224)) 

    game.init()
    elements = [1,1,1,1]
    while not game.is_episode_finished():
        action = [0] * 20
        s = game.get_state()
        frame = s.screen_buffer
        frame = torch.tensor(frame)
        frame = transform(frame)
        frame = frame.reshape(1,1,3,224,224).to(torch.float32).to(device)
        wasd_out, mouse_LR_out, mouse_UD_out, mouse_shoot_out  = model(frame)


        wasd_out = wasd_out.squeeze(0).squeeze(0)
        wasd_out = wasd_out.cpu().detach().numpy()
        samples = np.random.binomial(1,wasd_out )
        #sampled_element = random.choices(elements, weights= wasd_out, k=4)
        #print(sampled_element)
        action[3] = samples[0] #Right
        action[4] = samples[1] #left
        action[5] = samples[2] #back
        action[6] = samples[3] #forward
        
        print(action)
        mouse_shoot_out= mouse_shoot_out.squeeze(0).squeeze(0).cpu().detach().numpy()
        shoot = np.random.binomial(1,mouse_shoot_out)
        action[0] = shoot

        mouse_UD_out = mouse_UD_out.squeeze(0).squeeze(0).cpu().detach().numpy()
        mouse_LR_out = mouse_LR_out.squeeze(0).squeeze(0).cpu().detach().numpy()

        action[-3] = mouse_UD_out
        action[-2] = mouse_LR_out
    total_reward = game.get_total_reward()
    wandb.log({"episode total reward":total_reward })
    game.close()

    if total_reward > REWARD: 
        REWARD = total_reward
        train_state = {                
            'epoch': epoch,
            'state_dict': model.state_dict(),                
            'model': model,
                }
                
        torch.save(
            train_state,
            "/home/stannis/puren/model.pth"
            )
def train_one_epoch(model, data_loader,  n_epoch, opt, device):
    for ep in range(n_epoch):
        ep_loss = 0
        for imgs, actions in data_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            wasd_out, mouse_LR_out, mouse_UD_out, mouse_shoot_out  = model(imgs)

            wasd = actions[: , :, 0:4]
            shoot = actions[:, :, 0]
            mouse_UD = actions[:, : , -1]
            mouse_LR = actions[:, : , -2]

            loss = nn.BCELoss()
            #d_probs, d_actual = wasd_out[:, :, 1], wasd[:, :, 1]
            #d_bce_loss = torch.nn.functional.binary_cross_entropy(d_probs, d_actual)
            #a_probs, a_actual = wasd_out[:, :, 2], wasd[:, :, 2]
            #a_bce_loss = torch.nn.functional.binary_cross_entropy(a_probs, a_actual)
            wasd_bce_loss = loss(wasd_out, wasd)
            #wasd_bce_loss = wasd_bce_loss.mean(dim=1)

            mouse_shoot_out = mouse_shoot_out.squeeze(-1)

            shoot_loss = torch.nn.functional.binary_cross_entropy(mouse_shoot_out, shoot)
            
            mouse_UD_out = mouse_UD_out.squeeze(-1)
            mouse_UD_loss = torch.nn.functional.mse_loss(mouse_UD_out,mouse_UD)
            
            mouse_LR_out = mouse_LR_out.squeeze(-1)
            mouse_LR_loss = torch.nn.functional.mse_loss(mouse_LR_out,mouse_LR)
            
            total_loss = wasd_bce_loss + shoot_loss + mouse_UD_loss + mouse_LR_loss
            ep_loss += total_loss.item()
            opt.zero_grad() 
            total_loss.backward()
            opt.step()

            
        wandb.log({"loss": ep_loss})

        if ep % 2 == 0:
            play_one_episode(model, device, ep)
        
if __name__ == "__main__":
    model = BaseModel()
    df_train, df_test = get_train_val_test()

    train_data = VizDoomData(df_train)
    #val_data = VizDoomData(df_val, istest= True)
    test_data = VizDoomData(df_test, istest= True)
    train_loader = DataLoader(train_data, batch_size= 1, shuffle= True, num_workers=4, pin_memory=True, worker_init_fn= seed_worker)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #val_data.to(device)
    #test_data.to(device)
    wandb.login()
    run = wandb.init(
    project="vizdoom-il",
    config={
        "learning_rate": 0.0001,
        "epochs": 50,
    #,
)
    n_epoch = 50 
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    train_one_epoch(model,train_loader, n_epoch, optimizer, device)
