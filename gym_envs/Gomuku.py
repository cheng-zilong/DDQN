from .TicTacToe import TicTacToeEnv

def make_gomuku15(seed, **kwargs):
    env = TicTacToeEnv(board_size = 15, win_size = 5)
    from .AtariWrapper import TotalRewardWrapper
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env

def make_gomuku19(seed, **kwargs):
    env = TicTacToeEnv(board_size = 15, win_size = 5)
    from .AtariWrapper import TotalRewardWrapper
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env