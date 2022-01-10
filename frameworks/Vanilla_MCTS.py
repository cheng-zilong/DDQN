#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
from __future__ import annotations
from utils.LogAsync import logger
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
from gym_envs.TicTacToe import TicTacToeEnv
from gym import Env
from numba import njit

@njit
def ucb1(win_num, visit_num, p_visit_num, CONST_C):
    return win_num/visit_num + CONST_C * np.sqrt(np.log(p_visit_num)/visit_num)

class Node:
    CONST_C = np.sqrt(2)
    
    def __init__(self, player:int, action:int, state:np.array, action_list: list[int], parent_node: Node):
        self.player = player
        self.parent_node = parent_node
        self.child_list: list[Node] = []
        self.action = action
        self.state = state
        self.not_explored_action_list = action_list
        self.win_num = 0
        self.visit_num = 0 
        self.is_terminal = False
        self.terminal_winner = None

    def ucb1(self):
        if self.visit_num == 0:
            raise Exception('Node has not been visited!')
        return ucb1(self.win_num, self.visit_num, self.parent_node.visit_num, Node.CONST_C)

    def generate_child(self, player:int, action:int, state:np.array, action_list: list[int]):
        self.not_explored_action_list.remove(action)
        new_node = Node(player=player, action = action, state=state, action_list = action_list, parent_node=self)
        self.child_list.append(new_node)
        return new_node

    @property
    def win_rate(self):
        return self.win_num/self.visit_num

class Vanilla_MCTS:
    def __init__(self, env: Env, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.env = env
        self.simulator_env = deepcopy(self.env)

        self.root_node = Node(   player = (self.simulator_env.next_player+1)%2, 
                            action = None,
                            state = self.simulator_env.state, 
                            action_list = list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]),
                            parent_node = None)

    def _init_seed(self):
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])
    
    def selection(self):
        self.current_node = self.root_node
        while len(self.current_node.not_explored_action_list) == 0 and not self.current_node.is_terminal: # Not a leaf node
            # temp_max = -np.inf
            # for child in self.current_node.child_list:
            #     temp_ucb1 = child.ucb1()
            #     if temp_ucb1>temp_max:
            #          self.current_node = child
            #          temp_max = temp_ucb1
            ucb_list = [child.ucb1() for child in self.current_node.child_list]
            ucb_max = max(ucb_list)
            ucb_max_list = [self.current_node.child_list[i] for i in range(len(ucb_list)) if ucb_list[i] == ucb_max]
            self.current_node = random.choice(ucb_max_list)

    def expansion(self):
        if self.current_node.is_terminal:
            self.mcts_winner = self.current_node.terminal_winner
            return 
        self.simulator_env.reset_with_state(init_state = self.current_node.state)
        next_player = (self.current_node.player+1)%2
        action = random.sample(self.current_node.not_explored_action_list, 1)[0]
        new_state, _, _, _ = self.simulator_env.step(action)
        self.current_node = self.current_node.generate_child(player = next_player, 
                                                    action = action,
                                                    state = new_state,
                                                    action_list=list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]))
    
    def simulation(self):
        if self.current_node.is_terminal:
            return
        if self.simulator_env.status != 0: # draw or win
            self.current_node.is_terminal = True
            self.current_node.terminal_winner = self.simulator_env.winner
            self.mcts_winner = self.simulator_env.winner
            return
        temp = np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]
        temp = np.random.permutation(temp)
        action_list = list(temp)
        while True:
            random_action = action_list.pop()
            _, _, done, infos = self.simulator_env.step(random_action)
            # print("Infos : " + str(infos))
            # self.simulator_env.render()
            # print()
            if done: 
                self.mcts_winner = infos['winner']
                break

    def backup(self):
        while self.current_node is not None:
            self.current_node.visit_num += 1
            if self.mcts_winner == -1:
                self.current_node = self.current_node.parent_node
                continue
            if self.current_node.player == self.mcts_winner:
                self.current_node.win_num += 1
            else:
                self.current_node.win_num -= 1
            self.current_node = self.current_node.parent_node
    
    def search(self, iter_num):
        for _ in tqdm(range(iter_num+1)):
            self.selection()
            self.expansion()
            self.simulation()
            self.backup()

    def play(self, iter_num):
        self.search(iter_num = iter_num)
        visit_list = [child.visit_num for child in self.root_node.child_list]
        visit_max = max(visit_list)
        visit_max_list = [self.root_node.child_list[i] for i in range(len(visit_list)) if visit_list[i] == visit_max]
        chosen_node = random.choice(visit_max_list)
        print(f'\nWin Rate:{chosen_node.win_rate}')
        return chosen_node.action

    def choose_root(self, action):
        for child in self.root_node.child_list:
            if child.action == action:
                self.root_node = child 
                self.root_node.parent_node = None
                self.root_node.action = None
                return
        self.root_node = Node(   player = (self.env.next_player+1)%2, 
                            action = None,
                            state = self.env.state, 
                            action_list = list(np.asarray(range(self.env.action_space.n))[self.env.legal_action_mask]),
                            parent_node = None)

def play_with_me(board_size, win_size, is_AI_first=False, iter_num=10000, *args, **kwargs):
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    env.reset()
    env.render()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    monte = Vanilla_MCTS(env, *args, **kwargs)
    AI_player = 0 if is_AI_first else 1
    current_player = 0

    while not done:
        if current_player == AI_player:
            action = monte.play(iter_num=iter_num)
            print(f'AI turn:{action}')
        else:
            while True:
                try:
                    x, y = input("Your turn: ").split()
                    break
                except ValueError:
                    print('Illegal Input! Please indicate "row" and "column".')
            action = (int(x))*board_size+(int(y))
        _, _, done, infos = env.step(action)
        monte.choose_root(action)
        env.render()
        print(infos)
        current_player = (current_player+1)%2

# %%
