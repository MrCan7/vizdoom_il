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
    
    #game.set_doom_map("map01") #limited deathmatch
    # game.set_doom_map("map02") # full deathmatch. builtin map
    
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True) #may turn off
    #game.set_labels_buffer_enabled(True)
    #game.set_automap_buffer_enabled(True)

    game.set_sectors_info_enabled(True)
    game.set_render_crosshair(True)
    game.set_sound_enabled(True) #they mentioned there might be some issues on ubuntu 20.04. refer https://github.com/Farama-Foundation/ViZDoom/pull/486 for more details
    #game.set_audio_buffer_enabled(True)
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)

    #game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)
    #game.set_console_enabled(True)
    
    ## Set the game variables
    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    # +snd_efx 0 add this seperately for sound. no idea why
    DOOM_ENV_WITH_BOTS_ARGS_OLD = """
    -record multi_rec.lmp
    +snd_efx 0 
    +sv_respawnprotect 1
    +freelook 1
    -host 1 
    -deathmatch
    +timelimit 2.0 
    +sv_forcerespawn 1
    +sv_noautoaim 1
    +sv_spawnfarthest 1
    +sv_nocrouch 1 
    +viz_respawn_delay 1
    +viz_nocheat 1
    +name HumanPlayer 
    +colorset 0
    """
    
    DOOM_ENV_WITH_BOTS_ARGS = """
    -record multi_rec.lmp
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
    # DOOM_ENV_WITH_BOTS_ARGS = """
    # -host 1 
    # -deathmatch 
    # +viz_nocheat 0 
    # +cl_run 1 
    # +name AGENT 
    # +colorset 0 
    # +sv_forcerespawn 1 
    # +sv_respawnprotect 1 
    # +sv_nocrouch 1 
    # +sv_noexit 1
    # """

    '''
    game.add_game_args('-host 1'              # Game will start after one player connects (our bot).
                       'deathmatch'           # Game mode.
                       '+cl_run 1'            # By default the agent runs (instead of walking).
                       '+name AGENT' 
                       '+sv_forcerespawn 1'   # Agent automatically respawns.
                       '+sv_respawnprotect 1' # Couple of seconds on invicibility after respawning.
                       '+sv_nocrouch 1')      # Players can't crouch.
    '''
    
    game.add_game_args(DOOM_ENV_WITH_BOTS_ARGS)
    game.init()

    episodes = params["episodes"]
    num_bots = params["num_bots"]
    
    game.add_available_button(vzd.Button.LOOK_UP_DOWN_DELTA)

    print(game.get_available_buttons())
    for i in range(episodes):
        print(f"Episode #{i + 1}")
    
        #game.new_episode() 
        game.send_game_command("removebots")
        for i in range(num_bots):
            game.send_game_command("addbot")

        while not game.is_episode_finished():
            print(game.get_episode_timeout())
            print(game.get_episode_time())
            state = game.get_state()

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            if game.is_player_dead():
                print("Player died.")
                # Use this to respawn immediately after death, new state will be available.
                game.respawn_player()

            print(f"State #{state.number}")
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

        print("Episode finished!")
        print("FRAGS:" ,  game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        print("************************")
        game.new_episode()

        sleep(2.0)

    game.close()
