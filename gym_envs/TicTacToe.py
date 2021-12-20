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
        # player0 落子位置
        # Player1 落子位置
        # 全0表示现在是player0落子的回合，全1表示player1的回合
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.board_size, self.board_size), dtype=np.int8)
        self.reward_criterion = {'going':0, 'draw':0, 'win':1, 'loss':-1}
        # 用于检测胜负
        self.CONSTANT_A = 1 << np.arange(board_size*2)[::-1]

    def reset(self, init_state=None):
        if init_state is None:
            self.state = np.zeros([3,self.board_size,self.board_size], dtype=np.int8)  
            self.total_steps = 0
            self.status = 'going'
        else:
            self.state = np.array(init_state, dtype=np.int8, copy=True)
            self.total_steps = np.sum(np.asarray(init_state[0]+init_state[1]).reshape(-1) != 0)
            self.status = self.check_status() 
        next_player = self.state[2,0,0] 
        if self.status == 'win':
            self.winner = 0 if next_player == 1 else 1
            self.next_player = -1
        elif self.status == 'draw':
            self.winner = -1
            self.next_player = -1
        elif self.status == 'going':
            self.winner = -1
            self.next_player = next_player
        return self.state

    def check_status(self):
        if self.check_win():
            return 'win'
        if self.total_steps < self.board_size * self.board_size:
            return 'going'
        return 'draw'

    def check_win(self):
        #Vertical
        state_h = np.hstack(self.state[0:2]).dot(self.CONSTANT_A) #转成二进制
        for i in range(self.board_size - self.win_size+1):
            if reduce(and_, [state_h[i+j] for j in range(self.win_size)]):
                return True

        #Horizon
        state_v = self.CONSTANT_A.dot(np.vstack(self.state[0:2]))
        for i in range(self.board_size - self.win_size+1):
            if reduce(and_, [state_v[i+j] for j in range(self.win_size)]):
                return True

        # #diagnol 有问题
        # for i in range(self.board_size - self.win_size+1): #左移右移看对齐
        #     if reduce(and_, [state_h[i+j]<<j for j in range(self.win_size)]):
        #         return True
        #     if reduce(and_, [state_h[i+j]>>j for j in range(self.win_size)]):
        #         return True

        for p in range(2):
            for i in range(self.board_size - self.win_size + 1):
                for j in range(self.board_size - self.win_size + 1):
                    sub_matrix = self.state[p,i:self.win_size + i, j:self.win_size + j]
                    sub_matrix_diag1 = [sub_matrix[k][k] for k in range(self.win_size)]
                    sub_matrix_diag2 = [sub_matrix[self.win_size-1-k][k] for k in range(self.win_size)]
                    if (all(sub_matrix_diag1)  or all(sub_matrix_diag2)):
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
        current_player = self.next_player
        if self.status!='going':
            if self.winner == 0: # if player0 wins
                reward = [self.reward_criterion['win'], self.reward_criterion['loss']]
            elif self.winner == 1: # if player1 wins
                reward = [self.reward_criterion['loss'], self.reward_criterion['win']]
            else: # if draws
                reward = [self.reward_criterion['draw'], self.reward_criterion['draw']]
            done = True
        else:
            down_position = (int(action/self.board_size), int(action%self.board_size))
            if (self.state[0] + self.state[1])[down_position[0], down_position[1]] != 0:
                raise Exception('Illegal move')
            self.state[current_player, down_position[0], down_position[1]] = 1
            self.total_steps+=1
            if self.total_steps >= self.win_size * 2 - 1:
                # no one wins unless the number of steps is greater than win_size * 2 - 1
                self.status = self.check_status() 
            if self.status == 'win':
                self.winner = current_player
                self.next_player = -1
                if self.winner == 0: # if player0 wins
                    reward = [self.reward_criterion['win'], self.reward_criterion['loss']]
                elif self.winner == 1: # if player1 wins
                    reward = [self.reward_criterion['loss'], self.reward_criterion['win']]
                done = True
            elif self.status == 'draw':
                self.next_player = -1
                self.winner = -1
                reward = [self.reward_criterion['draw'], self.reward_criterion['draw']]
                done = True
            else:
                self.next_player = 0 if current_player == 1 else 1
                reward = [self.reward_criterion['going'], self.reward_criterion['going']]
                done = False
        return self.state, reward, done, {'steps':self.total_steps, 'winner':self.winner}

    def render(self, mode=None, close=False, folder=None, number=None):
        if folder == None:
            print('\nNext Player: %d'%(self.next_player))
            for i in range(self.board_size):
                print(" " + "-" * (self.board_size * 4 + 1))
                for j in range(self.board_size):
                    if self.state[0,i,j] == 1:
                        symbol = 'o'
                    elif self.state[1,i,j] == 1:
                        symbol = 'x'
                    else:
                         symbol = ' '
                    print(" | " + str(symbol), end='')
                print(" |")
            print(" " + "-" * (self.board_size * 4 + 1))
        else:
            file1 = open(folder + "/Render_" + ("" if number is None else str(number)) + ".txt", "a")
            file1.write('\nNext Player: %d'%(self.next_player))
            for i in range(self.board_size):
                file1.write(" " + "-" * (self.board_size * 4 + 1))
                for j in range(self.board_size):
                    if self.state[0,i,j] == 1:
                        symbol = 'o'
                    elif self.state[1,i,j] == 1:
                        symbol = 'x'
                    else:
                         symbol = ' '
                    file1.write(" | " + str(symbol), end='')
                file1.write(" |")
            file1.write(" " + "-" * (self.board_size * 4 + 1))
            file1.close()

def make_tic_tac_toe_env(**kwargs):
    env = TicTacToeEnv(board_size = kwargs['board_size'], win_size = kwargs['win_size'])
    from .AtariWrapper import TotalRewardWrapper
    env = TotalRewardWrapper(env)
    env.seed(kwargs['seed'])
    env.action_space.np_random.seed(kwargs['seed'])
    return env



# Test program 1, run 1 episode
# if __name__ == '__main__':
#     env = TicTacToeEnv(board_size=3, win_size=3) 
#     env.reset()
#     action_list = list(np.random.choice(9, 9, replace=False))
#     while True:
#         state, reward, done, infos = env.step(action_list.pop())
#         env.render()
#         print("Infos : " + str(infos))
#         print()
#         if done: 
#             env.reset()
#             break

# Test program 2, run many episodes
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    board_size = 3
    win_size = 3
    # board_size = 3
    # win_size = 3
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    env.reset()
    x_win_cnt = 0
    o_win_cnt = 0
    draw_cnt = 0
    for i in tqdm(range(50001)):
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