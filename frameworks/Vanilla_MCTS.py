
#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
from utils.LogAsync import logger
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
from gym_envs.TicTacToe import TicTacToeEnv
from utils.ActorAsync import ActorAsync

class Node:
    CONST_C = np.sqrt(2)
    def __init__(self, player, state, action, action_list:list, parent_node):
        self.player = player
        self.parent_node = parent_node
        self.child_list = []
        self.action = action
        self.state = state
        self.action_list = action_list 
        self.not_explored_action_list = action_list
        self.win_num = 0
        self.visit_num = 0 
        self.is_terminal = False

    def ucb1(self):
        if self.visit_num == 0:
            raise Exception('Node has not been visited!')
        return self.win_num/self.visit_num + Node.CONST_C * np.sqrt(np.log(self.parent_node.visit_num)/self.visit_num)

    def generate_child(self, player, state, action, action_list):
        self.not_explored_action_list.remove(action)
        new_node = Node(player=player, state=state, action = action, action_list = action_list, parent_node=self)
        self.child_list.append(new_node)
        return new_node

    @property
    def win_rate(self):
        return self.win_num/self.visit_num

class ActorAsync_MCTS(ActorAsync):
    def _eval(self, eval_idx, *args, **kwargs):
        return super()._eval(eval_idx, *args, **kwargs)

class Vanilla_MCTS:
    def __init__(self, env, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.env = env
        self.simulator_env = deepcopy(self.env)

        self.root = Node(   player = (self.simulator_env.next_player+1)%2, 
                            state = self.simulator_env.state, 
                            action = None,
                            action_list = list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]),
                            parent_node = None)

    def _init_seed(self):
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])
    
    def selection(self):
        self.current_node = self.root
        while len(self.current_node.not_explored_action_list) == 0 and not self.current_node.is_terminal: # Not a leaf node
            ucb_list = [child.ucb1() for child in self.current_node.child_list]
            ucb_max = max(ucb_list)
            ucb_max_list = [self.current_node.child_list[i] for i in range(len(ucb_list)) if ucb_list[i] == ucb_max]
            self.current_node = random.choice(ucb_max_list)

    def expansion(self):
        self.simulator_env.reset_with_state(init_state = self.current_node.state)
        if self.current_node.is_terminal:
            return 
        next_player = (self.current_node.player+1)%2
        action = random.sample(self.current_node.not_explored_action_list, 1)[0]
        new_state, _, _, _ = self.simulator_env.step(action)
        self.current_node = self.current_node.generate_child(player = next_player, 
                                                    action = action,
                                                    state = new_state,
                                                    action_list=list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]))
            
    def simulation(self):
        if self.simulator_env.status == 'win':
            self.current_node.is_terminal = True
            self.winner = self.simulator_env.winner
            return
        if self.simulator_env.status == 'draw':
            self.current_node.is_terminal = True
            self.winner = -1
            return
        temp = np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask]
        random.shuffle(temp)
        action_list = list(temp)
        while True:
            _, _, done, infos = self.simulator_env.step(action_list.pop())
            # print("Infos : " + str(infos))
            # self.simulator_env.render()
            # print()
            if done: 
                self.winner = infos['winner']
                break

    def backup(self):
        while self.current_node is not None:
            self.current_node.visit_num += 1
            if self.winner == -1:
                self.current_node = self.current_node.parent_node
                continue
            if self.current_node.player == self.winner:
                self.current_node.win_num += 1
            else:
                self.current_node.win_num -= 1
            self.current_node = self.current_node.parent_node

    def search(self, iter_num):
        for i in tqdm(range(iter_num+2)):
            self.selection()
            self.expansion()
            self.simulation()
            self.backup()
            if i == iter_num:
                bbb=1

    def play(self, iter_num):
        self.search(iter_num = iter_num)
        visit_list = [child.visit_num for child in self.root.child_list]
        visit_max = max(visit_list)
        visit_max_list = [self.root.child_list[i] for i in range(len(visit_list)) if visit_list[i] == visit_max]
        chosen_node = random.choice(visit_max_list)
        print(f'\nWin Rate:{chosen_node.win_rate}')
        return chosen_node.action

def play_with_me(*args, **kwargs):
    board_size = 3
    win_size = 3
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    env.reset()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    while not done:
        monte = Vanilla_MCTS(env, *args, **kwargs)
        ai_action = monte.play()
        print(f'AI turn:{ai_action}')
        _, _, done, infos = env.step(ai_action)
        env.render()
        print(infos)

        if done:
            break 

        my_action = int(input("Your turn:"))
        _, _, done, infos = env.step(my_action)
        env.render()
        print(infos)

def play_with_me2(*args, **kwargs):
    board_size = 10
    win_size = 5
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    env.reset()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    env.render()
    while not done:
        x, y = input("Your turn: ").split()
        _, _, done, infos = env.step((int(x))*board_size+(int(y)))
        env.render()
        print(infos)

        if done:
            break 

        monte = Vanilla_MCTS(env, *args, **kwargs)
        ai_action = monte.play(iter_num=10000)
        print(f'AI turn:{ai_action}')
        _, _, done, infos = env.step(ai_action)
        env.render()
        print(infos)

# %%
