import torch
from utils.Config import get_default_parser
from gym_envs.AtariWrapper import make_atari_env
from utils.Network import *
from utils.LogProcess import logger

# # Atari
if __name__ == '__main__':
    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.C51_DQN import C51_DQN_Async
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # kwargs['train_update_target_freq']=10000
    # kwargs['train_start_step'] = 50000
    # kwargs['actor_num']=1
    # kwargs['eps_decay_steps']=250000/kwargs['actor_num']
    # kwargs['actor_queue_buffer_size'] = 10
    # C51_DQN_Async(
    #     make_env_fun = make_atari_env,
    #     network_fun = CatCnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),
    #     **kwargs
    # ).train()

    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.Vanilla_DQN import Vanilla_DQN_Async   
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # kwargs['train_update_target_freq']=10000
    # kwargs['train_start_step'] = 50000
    # kwargs['actor_num']=1
    # kwargs['eps_decay_steps']=250000/kwargs['actor_num']
    # kwargs['actor_queue_buffer_size'] = 10
    # Vanilla_DQN_Async(
    #     make_env_fun = make_atari_env,
    #     network_fun = CnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),
    #     **kwargs
    # ).train()

    # from frameworks.Vanilla_MCTS import play_with_me 
    # play_with_me(   
    #     board_size=7, 
    #     win_size=5, 
    #     is_AI_first=False, 
    #     iter_num=10000, 
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
    # play_with_me(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220122-203908/40000.pt', True, **kwargs)
    # AI_play_again_AI(make_gomuku,AlphaZeroNetwork,'save_model/AlphaZero(TotalRewardWrapper)_4_20220122-203908/40000.pt', **kwargs)

    # kwargs = vars(get_default_parser().parse_args())
    # from frameworks.Vanilla_DDPG import Vanilla_DDPG_Async
    # from gym_envs.Mujoco import make_mujoco
    # kwargs['train_start_step'] = 10000
    # kwargs['train_update_tau'] = 0.005
    # kwargs['buffer_size'] = 1000000
    # kwargs['batch_size'] = 128 
    # kwargs['sigma_start'] = 0.1
    # kwargs['sigma_end'] = 0.1
    # kwargs['sigma_decay_steps'] = 1000000
    # kwargs['eval_sigma'] = 0.0
    # kwargs['env_name'] = 'Walker2d-v3' # Walker2d-v3 HalfCheetah-v3
    # kwargs['stack_frames'] = 1
    # kwargs['lr_actor'] = 1e-3
    # kwargs['lr_critic'] = 1e-3
    # kwargs['train_network_freq'] = 25
    # kwargs['actor_num'] = 2
    # kwargs['eval_freq'] = 20000
    # Vanilla_DDPG_Async(
    #     make_env_fun = make_mujoco,
    #     network_fun = LinearDDPGNetwork, 
    #     optimizer_fun = [
    #         lambda params: torch.optim.Adam(params, lr=kwargs['lr_actor'], eps=kwargs['optimizer_eps']),  
    #         lambda params: torch.optim.Adam(params, lr=kwargs['lr_critic'], eps=kwargs['optimizer_eps'])
    #     ],
    #     **kwargs
    # ).train()

    kwargs = vars(get_default_parser().parse_args())
    from frameworks.Vanilla_TD3 import Vanilla_TD3_Async
    from gym_envs.Mujoco import make_mujoco
    kwargs['train_start_step'] = 10000
    kwargs['train_update_tau'] = 0.005
    kwargs['buffer_size'] = 1000000
    kwargs['batch_size'] = 128 
    kwargs['sigma_start'] = 0.1
    kwargs['sigma_end'] = 0.1
    kwargs['sigma_decay_steps'] = 1000000
    kwargs['actor_noise'] = 0.2
    kwargs['noise_clip'] = 0.5
    kwargs['train_actor_freq'] = 2
    kwargs['eval_sigma'] = 0.0
    kwargs['env_name'] = 'Walker2d-v3'
    kwargs['stack_frames'] = 1
    kwargs['lr_actor'] = 1e-3
    kwargs['lr_critic'] = 1e-3
    kwargs['train_network_freq'] = 25
    kwargs['actor_num'] = 2
    kwargs['eval_freq'] = 20000
    Vanilla_TD3_Async(
        make_env_fun = make_mujoco,
        network_fun = LinearTD3Network, 
        optimizer_fun = [
            lambda params: torch.optim.Adam(params, lr=kwargs['lr_actor'], eps=kwargs['optimizer_eps']),  
            lambda params: torch.optim.Adam(params, lr=kwargs['lr_critic'], eps=kwargs['optimizer_eps'])
        ],
        **kwargs
    ).train()

