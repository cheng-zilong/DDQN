#%%
import gym
import numpy as np
from gym import spaces
from functools import reduce
from operator import and_

class TicTacToeEnv(gym.Env):
    def __init__(self, board_size, win_size):
        super(TicTacToeEnv, self).__init__()
        self.win_size = win_size
        self.board_size = board_size
        self.symbols = [' ', 'x', 'o']
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.rewards = {'going':0, 'draw':0, 'win':1, 'bad_position':-1}
        self.CONSTANT_A = 1 << np.arange(board_size*2)[::-1]

    def reset(self, init_state=None, next_player=None):
        if init_state is None:
            self.state = np.zeros(self.board_size * self.board_size, dtype=np.int8)  
            self.total_steps = 0
        else:
            self.state = np.array(init_state, dtype=np.int8, copy=True)
            self.total_steps = np.sum(np.asarray(init_state)!=0)
        self.status = self.check_status() 
        if self.status == 'win':
            self.winner = 1 if next_player != 1 else 2
            self.next_player = 0
        elif self.status == 'draw':
            self.winner = 0
            self.next_player = 0
        elif self.status == 'going':
            self.winner = 0
            self.next_player = 1 if next_player is None else next_player
        return self.state

    def check_status(self):
        if self.check_win():
            return 'win'
        if self.total_steps < self.board_size * self.board_size:
            return 'going'
        return 'draw'

    def check_win(self):
        state_list = [np.array(self.state == 1).reshape(self.board_size,-1), np.array(self.state == 2).reshape(self.board_size,-1)]
        #Vertical
        state_h = np.hstack(state_list).dot(self.CONSTANT_A) #转成二进制
        for i in range(self.board_size - self.win_size+1):
            if reduce(and_, [state_h[i+j] for j in range(self.win_size)]):
                return True

        #Horizon
        state_v = self.CONSTANT_A.dot(np.vstack(state_list))
        for i in range(self.board_size - self.win_size+1):
            if reduce(and_, [state_v[i+j] for j in range(self.win_size)]):
                return True

        #diagnol
        for i in range(self.board_size - self.win_size+1): #左移右移看对齐
            if reduce(and_, [state_h[i+j]<<j for j in range(self.win_size)]):
                return True
            if reduce(and_, [state_h[i+j]>>j for j in range(self.win_size)]):
                return True
        return False

        # for state in state_list:
        #     for i in range(self.board_size - self.win_size+1):
        #         if any(reduce(and_, [state[:, i+j] for j in range(self.win_size)])):
        #             return True

        # for i in range(0, self.board_size * self.board_size, self.board_size):
        #     cnt = 0
        #     k = i
        #     for j in range(1, self.board_size):
        #         (cnt, k) = (cnt + 1, k) if (self.state[k] == self.state[i + j] and self.state[k] != 0) else (0, i + j)
        #         if cnt == self.win_size - 1:
        #             return True

        # for i in range(0, self.board_size):
        #     cnt = 0
        #     k = i
        #     for j in range(self.board_size, self.board_size * self.board_size, self.board_size):
        #         (cnt, k) = (cnt + 1, k) if (self.state[k] == self.state[i + j] and self.state[k] != 0) else (0, i + j)
        #         if cnt == self.win_size - 1:
        #             return True

        # matrix = self.state.reshape(self.board_size,-1)
        # for i in range(self.board_size - self.win_size + 1):
        #     for j in range(self.board_size - self.win_size + 1):
        #         sub_matrix = matrix[i:self.win_size + i, j:self.win_size + j]
        #         sub_matrix_diag1 = [sub_matrix[k][k]==sub_matrix[0][0] for k in range(1,self.win_size)]
        #         sub_matrix_diag2 = [sub_matrix[self.win_size-1-k][k]==sub_matrix[self.win_size-1][0] for k in range(1,self.win_size)]
        #         if (all(sub_matrix_diag1) and (sub_matrix[0][0] != 0)) or (all(sub_matrix_diag2) and (sub_matrix[self.win_size-1][0] != 0)):
        #             return True
        return False

    def step(self, action):
        illegal_move = False
        if self.status!='going':
            return self.state, self.rewards[self.status], self.status!='going', {'steps':self.total_steps, 'illegal_move':illegal_move, 'next_player':self.next_player, 'winner':self.winner}
        if self.state[action] != 0:
            illegal_move = True
        else:
            self.state[action] = self.next_player
            self.total_steps+=1
            if self.total_steps >= self.win_size * 2 - 1:
                # noone wins unless the number of steps is greater than win_size * 2 - 1
                self.status = self.check_status() 
            if self.status == 'win':
                self.winner = self.next_player
                self.next_player = 0
            elif self.status == 'draw':
                self.next_player = 0
            else:
                self.next_player = 2 if self.next_player == 1 else 1
        return self.state, self.rewards[self.status], self.status!='going', {'steps':self.total_steps, 'illegal_move':illegal_move, 'next_player':self.next_player, 'winner':self.winner}

    def render(self, mode=None, close=False):
        grid = [self.symbols[value] for value in self.state]
        for j in range(0, self.board_size * self.board_size, self.board_size):
            print(" " + "-" * (self.board_size * 4 + 1))
            for i in range(self.board_size):
                print(" | " + str(grid[i + j]), end='')
            print(" |")
        print(" " + "-" * (self.board_size * 4 + 1))

if __name__ == '__main__':
    env = TicTacToeEnv(board_size=3, win_size=3) 
    env.reset()
    action_list = list(np.random.choice(9, 9, replace=False))
    while True:
        state, reward, done, infos = env.step(action_list.pop())
        print("Infos : " + str(infos))
        env.render()
        print()
        if done: 
            env.reset()
            break
        
#%%