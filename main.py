
import torch
from utils.Config import get_default_parser
from utils.LogAsync import logger
from gym_envs.AtariWrapper import make_atari_env
from utils.Network import *


# # Atari
if __name__ == '__main__':

    # from frameworks.C51_DQN import C51_DQN
    # kwargs = vars(get_default_parser().parse_args())
    # kwargs['train_start_step'] = 50000
    # kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    # C51_DQN(
    #     make_env_fun = make_atari_env,
    #     network_fun = CatCnnQNetwork, #CnnQNetwork CatCnnQNetwork
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    #     ).train()

    from frameworks.Nature_DQN import Nature_DQN   
    kwargs = vars(get_default_parser().parse_args())
    kwargs['train_start_step'] = 50000
    kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    Nature_DQN(
        make_env_fun = make_atari_env,
        network_fun = CnnQNetwork, #CnnQNetwork CatCnnQNetwork
        optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
        **kwargs
        ).train()

    # from gym_envs.TicTacToe import make_tic_tac_toe_env
    # from gym_envs.Gomuku import make_gomuku15
    # from frameworks.Nature_DQN_TwoPlayer import Nature_DQN_TwoPlayer  
    # kwargs['stack_frames'] = 1
    # kwargs['train_log_freq'] = 10
    # kwargs['ep_reward_avg_number'] = 10
    # kwargs['train_network_freq'] = 32
    # kwargs['clip_gradient'] = 100000
    # Nature_DQN_TwoPlayer(
    #     make_env_fun = make_tic_tac_toe_env, 
    #     network_fun = CnnQNetwork_TicTacToe,  
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']), 
    #     **kwargs
    #     ).train()