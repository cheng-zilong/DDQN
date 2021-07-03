import random
import torch
from utils.Network import *
from utils.Config import get_default_parser
import numpy as np
from utils.Wrapper import make_env
from utils.LogAsync import logger
from frameworks.DQN import CatDQN

if __name__ == '__main__':
    parser = get_default_parser()
    parser.set_defaults(seed=666) 
    # parser.set_defaults(env_name= 'BreakoutNoFrameskip-v4')
    parser.set_defaults(env_name= 'SpaceInvadersNoFrameskip-v4')
    # parser.set_defaults(env_name= 'PongNoFrameskip-v4')
    
    parser.set_defaults(eval_render_save_gif=[1]) # save 1 and 5
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.init(project_name='C51', args=args)

    if args.mode == 'train':
        CatDQN(
            make_env = make_env,
            network = CatCnnQNetwork, 
            optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).train()
    elif args.mode == 'eval':
        CatDQN(
            make_env = make_env,
            network = CatCnnQNetwork, 
            optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).eval()