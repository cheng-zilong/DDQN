import torch
from utils.Config import get_default_parser
from utils.LogAsync import logger
from gym_envs.AtariWrapper import make_atari_env
from utils.Network import *

# # Atari
if __name__ == '__main__':
    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.C51_DQN import C51_DQN
    # kwargs['train_start_step'] = 50000
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # C51_DQN(
    #     make_env_fun = make_atari_env,
    #     network_fun = CatCnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    # ).train()

    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.Nature_DQN import Nature_DQN   
    # kwargs['train_start_step'] = 50000
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # Nature_DQN(
    #     make_env_fun = make_atari_env,
    #     network_fun = CnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    # ).train()

    # kwargs = vars(get_default_parser().parse_args()) TODO GOMUKU is removed
    # from gym_envs.TicTacToe import make_tic_tac_toe_env
    # from gym_envs.Gomuku import make_gomuku8, make_gomuku15, make_gomuku19
    # from frameworks.Nature_DQN_Board import Nature_DQN_TwoPlayer_Gomuku  
    # kwargs['stack_frames'] = 1
    # kwargs['train_log_freq'] = 10
    # kwargs['ep_reward_avg_number'] = 10
    # kwargs['train_network_freq'] = 32
    # kwargs['clip_gradient'] = 100000
    # Nature_DQN_TwoPlayer_Gomuku(
    #     make_env_fun = make_gomuku8, 
    #     network_fun = CnnQNetwork_TicTacToe,  
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']), 
    #     **kwargs
    # ).train()

    # from frameworks.Vanilla_MCTS import play_with_me 
    # play_with_me(   
    #     board_size=7, 
    #     win_size=5, 
    #     is_AI_first=False, 
    #     iter_num=100000, 
    #     seed=1, 
    #     project_name='C51'
    # )

    kwargs = vars(get_default_parser().parse_args())
    from frameworks.AlphaZero import AlphaZero
    from gym_envs.TicTacToe import make_gomuku
    from utils.Network import AlphaZeroNetwork # CnnQNetwork #AlphaZeroNetwork
    from frameworks.AlphaZero import play_with_me, AI_play_again_AI
    kwargs['train_log_freq'] = 10
    kwargs['mcts_sim_num'] = 800
    kwargs['residual_num'] = 10
    kwargs['filters_num'] = 256
    kwargs['board_size'] = 7
    kwargs['win_size'] = 4
    kwargs['SGD_weight_decay'] = 1e-4
    kwargs['SGD_momentum'] = 0.9
    kwargs['lr'] = 0.002
    kwargs['lr_decay_step_size'] = int(5e4)
    kwargs['actors_num'] = 3
    kwargs['stack_frames'] = 1
    kwargs['train_start_buffer_size'] = int(10000)
    kwargs['buffer_size'] = int(1e5)
    kwargs['train_steps'] = int(9e5)
    kwargs['batch_size'] = 256
    AlphaZero(
        make_env_fun = make_gomuku,
        network_fun = AlphaZeroNetwork, #CnnQNetwork, #AlphaZeroNetwork,
        # optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),
        optimizer_fun = lambda params: torch.optim.SGD(params, lr=kwargs['lr'], weight_decay=kwargs['SGD_weight_decay'], momentum=kwargs['SGD_momentum']),
        **kwargs
    ).train()
    # play_with_me(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220114-063557/70000.pt', False, **kwargs) # TODO start with 4, 3 has error
    # AI_play_again_AI(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220114-063557/60000.pt', **kwargs)


