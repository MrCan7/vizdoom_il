import vizdoom as vzd
import os
from parameters import params

if __name__ == "__main__":

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
    #game.set_depth_buffer_enabled(True) #may turn off
    #game.set_labels_buffer_enabled(True)
    #game.set_automap_buffer_enabled(True)

    game.set_sectors_info_enabled(True)
    game.set_render_crosshair(True)
    #game.set_window_visible(True)
    game.set_sound_enabled(True) #they mentioned there might be some issues on ubuntu 20.04. refer https://github.com/Farama-Foundation/ViZDoom/pull/486 for more details
    #game.set_audio_buffer_enabled(True)
    #game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)

    #game.set_living_reward(-1)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.set_console_enabled(True)
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
    while not game.is_episode_finished():
        s = game.get_state()
        a = game.get_last_action()
        r = game.get_last_reward()
        game.advance_action()
        print(a)
    game.close()