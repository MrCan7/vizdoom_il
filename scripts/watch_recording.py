import vizdoom as vzd
import os


if __name__ == "__main__":

    recording_path = "/home/puren/İTÜ_DERSLER/AI/PROJECT/vizdoom_il/data/multi_rec.lmp"

    game = vzd.DoomGame()

    scenario = "cig.cfg"
    scenario_path = os.path.join(vzd.scenarios_path, scenario)

    game.load_config(scenario_path)



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
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_console_enabled(True)

    game.add_game_args("+snd_efx 0") #add this seperately for sound. no idea why
    game.add_game_args(
        "-host 1 -deathmatch +timelimit 2.0 "
        "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
        "+viz_respawn_delay 10 +viz_nocheat 1"
        
    )

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")
    game.add_game_args("-host 1 -deathmatch")
    game.init()

    game.replay_episode("multi_rec.lmp")
    while not game.is_episode_finished():
        s = game.get_state()

        game.advance_action()
        a = game.get_last_action()
        r = game.get_last_reward()
        print(a)
    game.close()