#%%
from typing import overload
import gym
import numpy as np
from gym import spaces
from functools import reduce
from operator import and_
from copy import deepcopy
import torch
from numba import njit

# @profile
@njit
def check_win(state, board_size, win_size, CONSTANT_A):
    #Vertical
    state = state.astype(np.float32)
    CONSTANT_AA = CONSTANT_A[-board_size:]
    state_h1 = (state[0]@CONSTANT_AA).reshape(-1).astype(np.int32)
    state_h2 = (state[1]@CONSTANT_AA).reshape(-1).astype(np.int32)
    state_h = state_h1 + state_h2 * (2**board_size)
    for i in range(board_size - win_size+1):
        res = state_h[i]
        for j in range(1,win_size):
            res = res & state_h[i+j]
        if res:
            return True

    # #Horizon
    state_v = np.concatenate((state[0], state[1]))
    state_v = (CONSTANT_A.reshape(-1)@state_v).astype(np.int32)
    for i in range(board_size - win_size+1):
        res = state_v[i]
        for j in range(1,win_size):
            res = res & state_v[i+j]
        if res:
            return True

    #diagnol 
    for i in range(board_size - win_size+1): #左移右移看对齐
        res1 = state_h1[i]
        res2 = state_h1[i]
        res3 = state_h2[i]
        res4 = state_h2[i]
        for j in range(1,win_size):
            res1 &= state_h1[i+j]>>j
            res2 &= state_h1[i+j]<<j
            res3 &= state_h2[i+j]>>j
            res4 &= state_h2[i+j]<<j
        if res1 or res2 or res3 or res4:
            return True
    return False

class TicTacToeEnv(gym.Env):
    def __init__(self, board_size, win_size):
        super().__init__()
        self.win_size = win_size
        self.board_size = board_size
        self.symbols = [' ', 'x', 'o']
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        # player0: 落子位置
        # Player1: 落子位置
        # next_player: 全0表示现在是player0落子的回合，全1表示player1的回合
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.board_size, self.board_size), dtype=np.int8)
        self.reward_criterion = {0:0, 1:0, 2:1, 3:-1}
        # self.CONSTANT_A = 1 << np.arange(board_size*2)[::-1]
        self.CONSTANT_A = (1 << np.arange(board_size*2)[::-1]).reshape(-1,1).astype(np.float32)
        self.status = None #0:going 1:draw 2:win
    
    def reset(self):
        self.state = np.zeros([3,self.board_size,self.board_size], dtype=np.int8)  
        self.total_steps = 0
        self.status = 0 # going
        self._legal_action_mask = (self.state[0]+self.state[1] == 0)
        self.winner = -1
        self.next_player = 0
        return self.state.copy()

    
    def reset_with_state(self, init_state):
        self.state = np.array(init_state, dtype=np.int8, copy=True)
        self.total_steps = np.sum(self.state[0:2])
        self.status = self.check_status() 
        self._legal_action_mask = (self.state[0]+self.state[1] == 0)
        if self.status == 2:
            self.winner = (self.state[2,0,0]  + 1)%2
        elif self.status == 1:
            self.winner = -1
        elif self.status == 0:
            self.winner = -1
        self.next_player = self.state[2,0,0] 
        return self.state.copy()

    def reset_with_state_n_action(self, init_state, action):
        '''
        must ensure that the state is not teminal state
        '''
        self.state = np.array(init_state, dtype=np.int8, copy=True)
        self.total_steps = np.sum(self.state[0:2])
        self.status = 0
        self._legal_action_mask = (self.state[0]+self.state[1] == 0)
        self.winner = -1
        self.next_player = self.state[2,0,0] 
        state, _, _, _ = self.step(action)
        return state
    
    def check_status(self):
        # if self.check_win():
        if check_win(self.state[0:2], self.board_size, self.win_size, self.CONSTANT_A):
            return 2
        if self.total_steps < self.board_size * self.board_size:
            return 0
        return 1

    def step(self, action):
        current_player = self.next_player
        if self.status!=0:
            if self.winner == 0: # if player0 wins
                reward = [self.reward_criterion[2], self.reward_criterion[3]]
            elif self.winner == 1: # if player1 wins
                reward = [self.reward_criterion[3], self.reward_criterion[2]]
            else: # if draws
                reward = [self.reward_criterion[1], self.reward_criterion[1]]
            done = True
        else:
            down_position_x = action//self.board_size
            down_position_y = action%self.board_size
            if self.state[0, down_position_x, down_position_y] or self.state[1, down_position_x, down_position_y]:
                raise Exception('Illegal move')
            self._legal_action_mask[down_position_x, down_position_y] = False
            self.state[current_player, down_position_x, down_position_y] = 1
            self.state[2] = (current_player + 1)%2
            self.total_steps+=1
            if self.total_steps >= self.win_size * 2 - 1:
                # no one wins unless the number of steps is greater than win_size * 2 - 1
                self.status = self.check_status() 
            if self.status == 2:
                self.winner = current_player
                self.next_player = (current_player+1)%2
                if self.winner == 0: # if player0 wins
                    reward = [self.reward_criterion[2], self.reward_criterion[3]]
                elif self.winner == 1: # if player1 wins
                    reward = [self.reward_criterion[3], self.reward_criterion[2]]
                done = True
            elif self.status == 1:
                self.next_player = (current_player+1)%2
                self.winner = -1
                reward = [self.reward_criterion[1], self.reward_criterion[1]]
                done = True
            else:
                self.next_player = (current_player+1)%2
                reward = [self.reward_criterion[0], self.reward_criterion[0]]
                done = False
        return self.state.copy(), reward, done, {'winner':self.winner}

    def render(self, mode=None, close=False):
        if mode == "human" or mode == None:
            print('Next Player: %d'%(self.next_player))
            print("    " ,end='') 
            for i in range(self.board_size):
                print(" %2d "%(i), end='')
            print('\n', end='')
            for i in range(self.board_size):
                print("    " + "-" * (self.board_size * 4 + 1))
                for j in range(self.board_size):
                    if self.state[0,i,j] == 1:
                        symbol = 'o'
                    elif self.state[1,i,j] == 1:
                        symbol = 'x'
                    else:
                            symbol = ' '
                    if (j==0):
                        print(" %2d | "%(i) + str(symbol), end='')
                    else:
                        print(" | " + str(symbol), end='')
                print(" |")
            print("    " + "-" * (self.board_size * 4 + 1))
        elif mode == "rgb_array":
            return (self.state[0] * 0.5) + (self.state[1] * 1)

    @property
    def legal_action_mask(self):
        return self._legal_action_mask.reshape(-1)


# # Test program 1, run 1 episode
# if __name__ == '__main__':
#     from itertools import compress
#     env = TicTacToeEnv(board_size=3, win_size=3) 
#     env.reset()
#     while True:
#         action = np.random.choice(list(compress(range(9),env.legal_action_mask.reshape(-1))), 1)
#         _, reward, done, infos = env.step(action)
#         env.render()
#         print("Infos : " + str(infos))
#         print(reward)
#         print()
#         if done: 
#             env.reset()
#             break

def make_gomuku(seed, board_size, win_size, *args, **kwargs):
    from gym_envs.AtariWrapper import TotalRewardWrapper
    env = TicTacToeEnv(board_size = board_size, win_size = win_size)
    env = TotalRewardWrapper(env)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    return env

# Test program 2, run many episodes
if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    # board_size = 10
    # win_size = 5
    board_size = 3
    win_size = 3
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    env.reset()
    x_win_cnt = 0
    o_win_cnt = 0
    draw_cnt = 0
    for i in tqdm(range(1,500001)):
        action_list = list(np.random.choice(board_size*board_size, board_size*board_size, replace=False))
        while True:
            state, reward, done, infos = env.step(action_list.pop())
            if done: 
                env.reset()
                if (infos['winner']==0):
                    x_win_cnt+=1
                elif (infos['winner']==1):
                    o_win_cnt+=1
                else:
                    draw_cnt+=1
                break
        if i%1000==0:
            tqdm.write("%d\t%d\t%d\t%d"%(i, x_win_cnt, o_win_cnt, draw_cnt))


#%%