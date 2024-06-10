import os
import glob
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
rgb_path = "/home/stannis/puren/dataset/vizdoom/dataset/rgb"
actions_path = "/home/stannis/puren/dataset/vizdoom/dataset/actions"

STACK_SECONDS = True

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text)]


def read_actions(data_path, idx):
    actions_data = next(os.walk(actions_path))[1]
    actions=[]
    path= os.path.join(actions_path, idx, "actions.txt")
    with open(path) as file:
        for line in file: 
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            line = line.split(",")
            line = [float(s) for s in line]
            basic_actions = [line[0], line[3], line[4], line[5], line[6], line[-2], line[-3]] #shoot, D,A,S,W, MOUSE L/R DELTA, MOUSE UP/DOWN DELTA
            actions.append(basic_actions)
    
    return actions
    '''
    for idx in actions_data:
        path= os.path.join(actions_path, idx, "actions.txt")
        actions[idx] = {}
        with open(path) as file:
            frame_num= 0
            for line in file: 
                line = line.replace("[", "")
                line = line.replace("]", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                line = line.split(",")
                line = [float(s) for s in line]
                basic_actions = [line[0], line[3], line[4], line[5], line[6], line[-2], line[-3]] #shoot, D,A,S,W, MOUSE L/R DELTA, MOUSE UP/DOWN DELTA
                actions[idx][frame_num] = basic_actions
                frame_num += 1
    '''

    return actions

def get_train_val_test():
    data = []
    episode_content =[]
    id_counter= 0 

    #actions = read_actions(actions_path)
    rgb_data = next(os.walk(rgb_path))[1]

    for idx in rgb_data: 
        curr_ep_path = os.path.join(rgb_path, idx)
        imgs = glob.glob(os.path.join(curr_ep_path, "*.jpeg"))
        imgs.sort(key=natural_keys)

        actions = read_actions(actions_path, idx)
        ep_id = os.path.basename(curr_ep_path)
        if STACK_SECONDS == True:
            stacked_secs = 1 
            for i in range(0, len(imgs)-(35*stacked_secs), stacked_secs*35):
                img = imgs[i: i+(stacked_secs*35)]
                action = actions[i : i+(stacked_secs*35)]

                data.append({
                "unique_id": id_counter,
                "episode_no": ep_id,
                "img_path" : img,
                "img_actions": action
                })
                id_counter +=1
        else:
            for i in range(len(imgs)):
                img = imgs[i]
                action = actions[i]


                episode_content.append({
                    "id": i, #not sure abt this
                    "img_path": img,
                    "img_actions": action
                })
                
            data.append({
                "unique_id": id_counter,
                "episode_no": ep_id,
                "episode_content": episode_content
            })
            id_counter +=1
        
        
    df = pd.DataFrame.from_dict(data)    
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    #df_train, df_val = train_test_split(df_train, test_size=0.1, shuffle=True, random_state=42)

    return df_train,df_test
if __name__ =="__main__":
    '''
    df_train, df_val, df_test = get_train_val_test()
    data = VizDoomData(df_train)
    print(data[1])
    '''
    pass