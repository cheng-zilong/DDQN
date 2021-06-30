import random
import torch
import torch.nn as nn
import torch.optim as optim
from Network import *
from collections import deque
import argparse
from Config import get_default_parser
import wandb
import numpy as np
import time
from wrapper import make_env
from statistics import mean
from ReplayBufferAsync import ReplayBufferAsync
from LogAsync import logger
import torch.multiprocessing as mp
from ActorAsync import ActorAsync
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from DQN import CatDQN

if __name__ == '__main__':
    parser = get_default_parser()
    parser.set_defaults(seed=666) 
    # parser.set_defaults(env_name= 'BreakoutNoFrameskip-v4')
    parser.set_defaults(env_name= 'SpaceInvadersNoFrameskip-v4')
    # parser.set_defaults(env_name= 'PongNoFrameskip-v4')
    parser.set_defaults(train_steps = int(5e7))
    
    parser.set_defaults(start_training_steps=50000)
    parser.set_defaults(eval_freq=1000000)
    # parser.set_defaults(start_training_steps=1000)
    parser.set_defaults(eval_number=5)
    parser.set_defaults(eval_render_save_gif=[1,5]) # save 1 and 5

    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.)
    parser.add_argument('--v_max', type=float, default=10.)
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