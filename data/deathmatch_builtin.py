import vizdoom as vzd 
import os 
from parameters import params
from time import sleep

if __name__ == "__main__":
    game = vzd.DoomGame()
    ## Load the scenario
    # Dirty way to get the path to the scenario folder
    current_file_path = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file_path)
    scenario_directory = os.path.dirname(current_directory) + "/scenarios"
    
    scenario = params["scenario"] + ".cfg"
    scenario_path = os.path.join(scenario_directory, scenario)
    game.load_config(scenario_path)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    #game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True) #may turn off
    game.set_render_crosshair(True)
    game.set_sound_enabled(True) #they mentioned there might be some issues on ubuntu 20.04. refer https://github.com/Farama-Foundation/ViZDoom/pull/486 for more details
    #game.set_audio_buffer_enabled(True)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.add_game_args("+freelook 1")
    #game.add_game_args("timelimit 2.0")
    game.init()

    episodes = params["episodes"]
    for i in range(episodes):

        # new_episode can record the episode using Doom's demo recording functionality to given file.
        # Recorded episodes can be reconstructed with perfect accuracy using different rendering settings.
        # This can not be used to record episodes in multiplayer mode.
        game.new_episode(f"episode{i}_rec.lmp")

        while not game.is_episode_finished():
            s = game.get_state()

            state = game.get_state()

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            print(f"State #{s.number}")
            print("Game variables:", s.game_variables[0])
            print("=====================")

        print(f"Episode {i} finished. Saved to file episode{i}_rec.lmp")
        print("Total reward:", game.get_total_reward())
        print("************************\n")

game.new_episode()  # This is currently required to stop and save the previous recording.
game.close()