import torch
from utils.Config import get_default_parser
from utils.LogProcess import logger
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
    # from frameworks.Nature_DQN import Nature_DQN_Sync
    # kwargs['train_start_step'] = 50000
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # Nature_DQN_Sync(
    #     make_env_fun = make_atari_env,
    #     network_fun = CnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    # ).sync_train()

    kwargs = vars(get_default_parser().parse_args())
    from frameworks.Nature_DQN import Nature_DQN_Async   
    kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    kwargs['train_update_target_freq']=10000
    kwargs['train_start_step'] = 50000
    kwargs['actor_num']=2
    kwargs['eps_decay_steps']=250000/kwargs['actor_num']

    Nature_DQN_Async(
        make_env_fun = make_atari_env,
        network_fun = CnnQNetwork, #CnnQNetwork CatCnnQNetwork
        optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),
        **kwargs
    ).async_train()

    # from frameworks.Vanilla_MCTS import play_with_me 
    # play_with_me(   
    #     board_size=7, 
    #     win_size=5, 
    #     is_AI_first=False, 
    #     iter_num=100000, 
    #     seed=1, 
    #     project_name='C51'
    # )

    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.AlphaZero import AlphaZero
    # from gym_envs.TicTacToe import make_gomuku
    # from utils.Network import AlphaZeroNetwork # CnnQNetwork #AlphaZeroNetwork
    # from frameworks.AlphaZero import play_with_me, AI_play_again_AI
    # kwargs['train_log_freq'] = 10
    # kwargs['mcts_sim_num'] = 800
    # kwargs['residual_num'] = 10
    # kwargs['filters_num'] = 256
    # kwargs['board_size'] = 7
    # kwargs['win_size'] = 4
    # kwargs['SGD_weight_decay'] = 1e-4
    # kwargs['SGD_momentum'] = 0.9
    # kwargs['lr'] = 0.002
    # kwargs['lr_decay_step_size'] = int(1e5)
    # kwargs['lr_decay_gamma'] = 0.1
    # kwargs['actors_num'] = 4
    # kwargs['stack_frames'] = 1
    # kwargs['train_start_buffer_size'] = int(1e5)
    # kwargs['buffer_size'] = int(1e6)
    # kwargs['train_steps'] = int(9e5)
    # kwargs['batch_size'] = 128
    # AlphaZero(
    #     make_env_fun = make_gomuku,
    #     network_fun = AlphaZeroNetwork, #CnnQNetwork, #AlphaZeroNetwork,
    #     # optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),
    #     optimizer_fun = lambda params: torch.optim.SGD(params, lr=kwargs['lr'], weight_decay=kwargs['SGD_weight_decay'], momentum=kwargs['SGD_momentum']),
    #     **kwargs
    # ).train()
    # play_with_me(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220114-063557/70000.pt', False, **kwargs) # TODO start with 4, 3 has error
    # AI_play_again_AI(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220114-063557/60000.pt', **kwargs)


