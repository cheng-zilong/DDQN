from .TicTacToe import TicTacToeEnv
from .AtariWrapper import TotalRewardWrapper

def make_gomuku8(seed, *args, **kwargs):
    env = TicTacToeEnv(board_size = 8, win_size = 5)
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env

def make_gomuku15(seed, *args, **kwargs):
    env = TicTacToeEnv(board_size = 15, win_size = 5)
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env

def make_gomuku19(seed, *args, **kwargs):
    env = TicTacToeEnv(board_size = 19, win_size = 5)
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env