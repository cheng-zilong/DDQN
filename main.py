import random
import torch
from utils.Network import *
from utils.Config import get_default_parser
import numpy as np
from gym_envs.AtariWrapper import make_atari_env
from utils.LogAsync import logger
from frameworks.C51_DQN import C51_DQN
from frameworks.Nature_DQN import Nature_DQN
from frameworks.PlayAgainstSelf_DQN import PlayAgainstSelf_DQN
from gym_envs.TicTacToe import make_tic_tac_toe_env

# # Atari
if __name__ == '__main__':
    parser = get_default_parser()
    parser.set_defaults(seed=555) 
    parser.set_defaults(env_name= 'BreakoutNoFrameskip-v4')
    parser.set_defaults(start_training_steps= 10000)
    # parser.set_defaults(env_name= 'SpaceInvadersNoFrameskip-v4')
    # parser.set_defaults(env_name= 'PongNoFrameskip-v4')
    
    parser.set_defaults(eval_render_save_video=[1]) # save 1 and 5
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.init(project_name='C51', args=args)

    # if args.mode == 'train':
    #     C51_DQN(
    #         make_env_fun = make_atari_env,
    #         network_fun = CatCnnQNetwork, 
    #         optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
    #         **vars(args)
    #         ).train()
    # elif args.mode == 'eval':
    #     C51_DQN(
    #         make_env_fun = make_atari_env,
    #         network_fun = CatCnnQNetwork, 
    #         optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
    #         **vars(args)
    #         ).eval()

    if args.mode == 'train':
        Nature_DQN(
            make_env_fun = make_atari_env,
            network_fun = CnnQNetwork, 
            optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).train()
    elif args.mode == 'eval':
        Nature_DQN(
            make_env_fun = make_atari_env,
            network_fun = CnnQNetwork, 
            optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).eval()

# # # tic-tac-toe
# if __name__ == '__main__':
#     parser = get_default_parser()
#     parser.set_defaults(seed=555) 
#     parser.set_defaults(eval_render_save_video=[1]) # not save
#     parser.set_defaults(eval_display=True) # not save
#     args = parser.parse_args()
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     logger.init(project_name='AlphaGo', args=args)

#     if args.mode == 'train':
#         PlayAgainstSelf_DQN(
#             board_size=7,
#             win_size=5,
#             make_env_fun = make_tic_tac_toe_env,
#             network_fun = LinearQNetwork, 
#             optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
#             **vars(args)
#             ).train()
#     elif args.mode == 'eval':
#         PlayAgainstSelf_DQN(
#             board_size=7,
#             win_size=5,
#             make_env_fun = make_tic_tac_toe_env,
#             network_fun = LinearQNetwork, 
#             optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
#             **vars(args)
#             ).eval()