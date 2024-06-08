import vizdoom as vzd
import os
import numpy as np
from parameters import params
from PIL import Image
import cv2
if __name__ == "__main__":

    doom_red_color = [0, 0, 203]
    doom_blue_color = [203, 0, 0]

    def draw_bounding_box(buffer, x, y, width, height, color):
        """
        Draw a rectangle (bounding box) on a given buffer in the given color.
        """
        for i in range(width):
            buffer[y, x + i, :] = color
            buffer[y + height, x + i, :] = color

        for i in range(height):
            buffer[y + i, x, :] = color
            buffer[y + i, x + width, :] = color

    def color_labels(labels):
        """
        Walls are blue, floor/ceiling are red (OpenCV uses BGR).
        """
        tmp = np.stack([labels] * 3, -1)
        tmp[labels == 0] = [255, 0, 0]
        tmp[labels == 1] = [0, 0, 255]

        return tmp

    recording_path = "/home/puren/İTÜ_DERSLER/AI/PROJECT/vizdoom_il/data/multi_rec.lmp"

    game = vzd.DoomGame()

    ## Load the scenario
    # Dirty way to get the path to the scenario folder
    current_file_path = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file_path)
    scenario_directory = os.path.dirname(current_directory) + "/scenarios"
    
    scenario = params["scenario"] + ".cfg"
    scenario_path = os.path.join(scenario_directory, scenario)
    game.load_config(scenario_path)
    
    if scenario == "cig.cfg":
        game.set_doom_map("map02")

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    #game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True) #may turn off
    game.set_labels_buffer_enabled(True)
    #game.set_automap_buffer_enabled(True)

    game.set_sectors_info_enabled(True)
    game.set_render_crosshair(True)
    #game.set_window_visible(True)
    game.set_sound_enabled(True) #they mentioned there might be some issues on ubuntu 20.04. refer https://github.com/Farama-Foundation/ViZDoom/pull/486 for more details
    game.set_audio_buffer_enabled(True)
    #game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)

    #game.set_living_reward(-1)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.set_console_enabled(True)
    game.set_render_all_frames(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    '''
    DOOM_ENV_WITH_BOTS_ARGS = """
    -host 1 
    -deathmatch
    +viz_nocheat 0 
    +cl_run 1 
    +name HumanPlayer 
    +colorset 0 
    +sv_forcerespawn 1 
    +sv_respawnprotect 1 
    +sv_nocrouch 1 
    +sv_noexit 0 
    +snd_efx 0 
    +freelook 1
    +timelimit 2.0 
    +sv_noautoaim 1
    +sv_spawnfarthest 1
    +viz_respawn_delay 1
    +viz_nocheat 1
    """
    
    game.add_game_args(DOOM_ENV_WITH_BOTS_ARGS)
    '''
    game.init()

    game.replay_episode("episode0_rec.lmp")
    if params["sample_recording"] == True: 
        frame_num = 0
        actions_file = open("actions.txt", "w")
        rewards_file = open("rewards.txt", "w")
    while not game.is_episode_finished():
        s = game.get_state()
        a = game.get_last_action()
        r = game.get_last_reward()

   
        if params["sample_recording"] == True: 
            
            #save rgb frames
            frame = s.screen_buffer
            img_rgb = np.transpose(frame, (1, 2, 0))  # Convert from CHW to HWC format
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            '''
            cv2.imwrite(f"./dataset/2/frame_{frame_num}.jpeg", img_bgr)
            
            #save depth frames 
            depth_frame = s.depth_buffer
            cv2.imwrite(f"./dataset/depth/2/frame_{frame_num}.png", depth_frame)

            #save actions
            actions_file.write(str(a)+"\n")

            #save rewards
            rewards_file.write(str(r)+"\n")
            '''
            #save label buffer
            screen = s.screen_buffer
            print(screen.shape)
            
            labels_buff = s.labels_buffer
            labels = s.labels
            for label in labels:
                draw_bounding_box(
                    img_bgr,
                    label.x,
                    label.y,
                    label.width,
                    label.height,
                    doom_blue_color,
                )     
            
            cv2.imwrite(f"./dataset/labels/2/frame_{frame_num}.jpeg", img_bgr)
            frame_num +=1
        game.advance_action()
    
    actions_file.close()
    rewards_file.close()
    game.close()