
import torch
from utils.Config import get_default_parser
from utils.LogAsync import logger
from gym_envs.AtariWrapper import make_atari_env
from utils.Network import *


# # Atari
if __name__ == '__main__':

    kwargs = vars(get_default_parser().parse_args())
    kwargs['start_training_steps'] = 50000
    kwargs['env_name'] = 'SpaceInvadersNoFrameskip-v4' #BreakoutNoFrameskip SpaceInvadersNoFrameskip PongNoFrameskip
    kwargs['seed']=555 
    

    # from frameworks.C51_DQN import C51_DQN
    # kwargs['policy_class']= CatCnnQNetwork #CnnQNetwork CatCnnQNetwork
    # logger.init(project_name='C51', **kwargs)
    # C51_DQN(
    #     make_env_fun = make_atari_env,
    #     network_fun = kwargs['policy_class'], 
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    #     ).train()

    from frameworks.Nature_DQN import Nature_DQN   
    kwargs['policy_class']= CnnQNetwork #CnnQNetwork CatCnnQNetwork
    logger.init(project_name='C51', **kwargs)
    Nature_DQN(
        make_env_fun = make_atari_env,
        network_fun = kwargs['policy_class'], 
        optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
        **kwargs
        ).train()

    # from gym_envs.TicTacToe import make_tic_tac_toe_env
    # from frameworks.Nature_DQN_TwoPlayer import Nature_DQN_TwoPlayer  
    # kwargs['policy_class'] = CnnQNetwork_TicTacToe
    # kwargs['stack_frames'] = 1
    # kwargs['train_log_freq'] = 1000
    # kwargs['ep_reward_avg_number'] = 100
    # logger.init(project_name='C51', **kwargs)
    # Nature_DQN_TwoPlayer(
    #     make_env_fun = make_tic_tac_toe_env,
    #     network_fun = kwargs['policy_class'], 
    #     optimizer_fun = lambda params: torch.optim.Adam(params, lr=kwargs['lr'], eps=kwargs['optimizer_eps']),  
    #     **kwargs
    #     ).train()

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
#             optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.optimizer_eps),  
#             **vars(args)
#             ).train()
#     elif args.mode == 'eval':
#         PlayAgainstSelf_DQN(
#             board_size=7,
#             win_size=5,
#             make_env_fun = make_tic_tac_toe_env,
#             network_fun = LinearQNetwork, 
#             optimizer_fun = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.optimizer_eps),  
#             **vars(args)
#             ).eval()