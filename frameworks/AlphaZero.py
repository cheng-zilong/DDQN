from __future__ import annotations
from multiprocessing import Value
from matplotlib.pyplot import tick_params
import torch
import torch.nn as nn
from gym_envs.TicTacToe import TicTacToeEnv
from utils.Network import *
from collections import deque
import time
from statistics import mean
from utils.ReplayBufferAsync import ReplayBufferAsync
from utils.LogAsync import logger
import torch.multiprocessing as mp
from utils.ActorAsync import NetworkActorAsync
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
from utils.ActorAsync import ActorAsync
from numba import njit
from torch import optim
from torch.multiprocessing import Value
import os
class AlphaZero:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.env = make_env_fun(*args, **kwargs)

        kwargs['policy_class'] = 'AlphaZero'
        kwargs['env_name'] = self.env.__class__.__name__
        kwargs['project_name'] = 'AlphaZero'
        logger.init(*args, **kwargs)

        self.network_lock = mp.Lock()
        self.replay_buffer = ReplayBufferAsync(*args, **kwargs)
        self.mcts_actors_list:list[AlphaZeroActorAsync] = []
        self.actors_num = kwargs['actors_num']
        
        for i in range(self.actors_num):
            kwargs['seed']+=i
            self.mcts_actors_list.append(AlphaZeroActorAsync(  
                make_env_fun=make_env_fun, 
                network_lock=self.network_lock, 
                replay_buffer=self.replay_buffer,
                *args, **kwargs
            ))
            self.mcts_actors_list[-1].start()
        self.replay_buffer.start()

        self.network = network_fun(self.env.observation_space.shape, self.env.action_space.n, *args, **kwargs).cuda().share_memory() 
        self.optimizer = optimizer_fun(self.network.parameters()) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.1) #TODO add to config

    def _init_seed(self):
        torch.manual_seed(self.kwargs['seed'])
        torch.cuda.manual_seed(self.kwargs['seed'])
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])

    def train(self):
        self.network.train()
        self.replay_buffer.init_data_example(
            action=np.ones(self.env.action_space.n, dtype=np.float32), 
            obs=self.env.reset(),
            reward=1,
            done=True
        )
        for i in range(self.actors_num):
            self.mcts_actors_list[i].collect(self.network)
        
        while self.replay_buffer.check_size() < self.kwargs['train_start_buffer_size']:
            logger.add({
                'main_net_mode': str(self.network.training),
                'replay_buffer_size':    str(self.replay_buffer.check_size())
            })
            logger.wandb_print('(Training)', step=0)
            time.sleep(10)

        mse_loss = nn.MSELoss()
        log_soft_max = nn.LogSoftmax(dim=1)
        for train_idx in range(self.kwargs['train_steps']):
            state, action, reward, _, _ = self.replay_buffer.sample()
            p, v = self.network(state)
            value_loss = mse_loss(v.squeeze(), reward) 
            policy_loss = -torch.sum(action*log_soft_max(p) - action*torch.log(action+1e-10), dim=1).mean()
            total_loss = value_loss + policy_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            with self.network_lock:
                self.optimizer.step()
                self.scheduler.step()
            
            if train_idx%self.kwargs['train_log_freq'] == 0:
                logger.add({
                    'main_net_mode': str(self.network.training),
                    'replay_buffer_size': self.replay_buffer.check_size(),
                    'value loss': value_loss.item(),
                    'policy loss': policy_loss.item(),
                    'total loss': total_loss.item()
                })
                logger.wandb_print('(Main Thread)', step=train_idx)

            if train_idx % 10000 == 0:
                if not os.path.exists('save_model/' + logger._run_name + '/'):
                    os.makedirs('save_model/' + logger._run_name + '/')
                torch.save(self.network.state_dict(), 'save_model/' + logger._run_name + '/' + str(train_idx) +'.pt')
    
class Node:
    def __init__(self,  player: int, action: int, prior_prob: float, parent_node: Node):
        self.player = player
        self.action = action

        self.parent_node = parent_node
        self.child_list:list[Node] = []

        self.visit_num_N = 0 
        self.win_num_W = 0
        self.mean_win_num_Q = 0
        self.prior_prob_P = prior_prob

        self.is_terminal = False

    def choose_and_update(
            self,   
            state: np.array, 
            value:float, 
            legal_action_list:list[int], 
            legal_action_prior_list:list[float],
            is_terminal=False,
            terminal_winner=-1
        ):
        self.state = state
        self.leaf_node_value = value # if this node is a leaf node, then the value is 
        self.legal_action_list = legal_action_list
        self.legal_action_prior_list = legal_action_prior_list
        self.is_terminal = is_terminal
        self.terminal_winner = terminal_winner

    def generate_child(self,    
            player: int, 
            action: int, 
            prior_prob: float
        ):
        new_node = Node(    
            player=player, 
            action=action, 
            prior_prob=prior_prob, 
            parent_node=self
        )
        self.child_list.append(new_node)
        return new_node

@njit
def puct(p_visit_num:int, visit_num:int, prior_p:float, mean_win_num_Q:float) -> float:
    CONST_C_BASE = 19652 
    CONST_C_INIT = 2.5 
    weighting_C_s = np.log((1+p_visit_num+CONST_C_BASE)/CONST_C_BASE) + CONST_C_INIT
    U_s_a = weighting_C_s*prior_p*np.sqrt(p_visit_num)/(1+visit_num)
    return mean_win_num_Q + U_s_a

class AlphaZeroActorAsync(ActorAsync):
    TotalActorNumber=0
    game_counter = Value('i', 0)
    def __init__(self, make_env_fun, replay_buffer:ReplayBufferAsync, network_lock, *args, **kwargs):
        super().__init__(make_env_fun=make_env_fun, *args, **kwargs)
        self._network_lock = network_lock
        self.replay_buffer = replay_buffer
        self.simulator_env:TicTacToeEnv = deepcopy(self.env)
        self.actor_id = AlphaZeroActorAsync.TotalActorNumber
        AlphaZeroActorAsync.TotalActorNumber+=1
        
    def collect(self, network, *args, **kwargs):
        kwargs['network'] = network
        self.send(self.COLLECT, (args, kwargs))

    def _collect(self, network:AlphaZeroNetwork):
        game_line = []
        #[(player, action, state]), ... ]
        self.shared_network = network 
        self.network = deepcopy(self.shared_network)
        self.network.eval()
        self.softmax = nn.Softmax(dim=0)
        mcts_time_list = [0]
        game_idx = 0
        game_tic = time.time()
        self.epi_step_idx = 0
        while True:
            if self.actor_done_flag:
                self.update_network()
                game_time = time.time() - game_tic
                game_tic = time.time()
                with AlphaZeroActorAsync.game_counter.get_lock():
                    AlphaZeroActorAsync.game_counter.value += 1
                    game_idx = AlphaZeroActorAsync.game_counter.value 
                logger.add({
                    'game_count': game_idx,
                    'mcts_time':mean(mcts_time_list),
                    'game_time':game_time,
                    'game_steps':self.epi_step_idx
                })

                # logger.terminal_print(  caption = '------------(Actor Thread %d)'%self.actor_id, 
                #                         log_dict_tmp={
                #                             '------------game_idx': str(game_idx),
                #                             '------------network_mode': str(self.network.training),
                #                             '------------mcts_time': str(mean(mcts_time_list)),
                #                             '------------game_time': str(game_time),
                #                             '------------game_steps': str(self.epi_step_idx)
                #                         })

                game_idx+=1
                self.epi_step_idx = 0
                game_line_size = len(game_line)
                for idx, (player, action, state) in enumerate(game_line):
                    if info['winner'] == -1 or action is None:
                        reward = 0
                    elif info['winner'] == player:
                        reward = 1
                    else:
                        reward = -1
                    self.replay_buffer.add(action, state, reward, False if idx!=game_line_size-1 else True)
                self.actor_state = self.env.reset()
                self.init_root_node()
                self.actor_done_flag = False 
                game_line = [(self.root_node.player, None, self.actor_state)]
                mcts_time_list = []
            mcts_tic = time.time()
            action, action_prob = self.mcts()
            mcts_time_list.append(time.time()-mcts_tic)
            self.actor_state, _, self.actor_done_flag, info = self.env.step(action)
            game_line.append(((self.env.next_player+1)%2, action_prob, self.actor_state))
            self.choose_root_node(action)
            self.epi_step_idx+=1

    def update_network(self):
        with self._network_lock:
            self.network.load_state_dict(self.shared_network.state_dict())
    
    # @profile
    def mcts(self):
        # for i in tqdm(range(self.kwargs['mcts_sim_num'])):
        for i in range(self.kwargs['mcts_sim_num']):
            self._select()
            self._expand_and_envaluate()
            self._backup()
        # logger.terminal_print(caption = '------------(Actor Thread %d)'%self.actor_id, 
        #         log_dict_tmp={'------------ep_step': str(self.epi_step_idx)}
        #     )

        init_prob_array = np.zeros(self.simulator_env.action_space.n, dtype=np.float32)
        dirichlet_array = np.random.dirichlet(np.ones(len(init_prob_array))*0.03).astype(np.float32) #TODO add 0.03 config
        final_prob_array = np.zeros(len(self.root_node.child_list), dtype=np.float32)
        for child_idx, child in enumerate(self.root_node.child_list):
            init_prob_array[child.action] = child.visit_num_N/(self.root_node.visit_num_N-1) # because the root node is chosen one more time
            final_prob_array[child_idx] = 0.75*init_prob_array[child.action] + 0.25*dirichlet_array[child_idx] # TODO add 0.75 and 0.25 config
        chosen_child = random.choices(self.root_node.child_list, weights = final_prob_array)[0]
        return chosen_child.action, init_prob_array #TODO THIS  

    # @profile
    def _select(self):
        self.current_node = self.root_node 
        while len(self.current_node.child_list) != 0: # Not a leaf node 
            max_puct = -inf
            for child in self.current_node.child_list:
                temp_puct = puct(child.parent_node.visit_num_N, child.visit_num_N, child.prior_prob_P, child.mean_win_num_Q)
                if temp_puct > max_puct:
                    max_puct = temp_puct
                    self.current_node = child
                    # Now the current node is the child node with the largest puct
        if self.current_node.is_terminal:
            return 
        if self.current_node.parent_node is None: # if this is the init state, it has been chosen and update
            return
        
        # Now a leaf node is chosen, because the leaf node is initlized with only the plyer, action, and prior prob, generate its state firstly
        new_state = self.simulator_env.reset_with_state_n_action(self.current_node.parent_node.state, self.current_node.action)
        # Evaluate the leaf node
        # with self._network_lock, torch.no_grad(): #TODO slow
        with torch.no_grad():
            p, v = self.network([new_state])
            p = p.view(-1)
            p[~self.simulator_env.legal_action_mask] = -inf # the illegal moves cannot be chosen
            p = self.softmax(p).cpu().numpy()
            v = v.item()
        # Update the leaf node
        legal_action_list = list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask])
        self.current_node.choose_and_update(
            state=new_state, 
            value=v,
            legal_action_list=legal_action_list,
            legal_action_prior_list=p,
            is_terminal=(self.simulator_env.status != 0),
            terminal_winner=self.simulator_env.winner
        ) 

    # @profile
    def _expand_and_envaluate(self):
        if self.current_node.is_terminal:
            return 
        for action in self.current_node.legal_action_list:
            self.current_node.generate_child(   
                player=(self.current_node.player+1)%2, 
                action=action,
                prior_prob=self.current_node.legal_action_prior_list[action]
            )

    # @profile
    def _backup(self):
        leaf_node = self.current_node
        while self.current_node is not None:
            self.current_node.visit_num_N += 1
            if self.current_node.player == leaf_node.player:
                self.current_node.win_num_W += leaf_node.leaf_node_value
            else:
                self.current_node.win_num_W -= leaf_node.leaf_node_value
            self.current_node.mean_win_num_Q = self.current_node.win_num_W/self.current_node.visit_num_N
            self.current_node = self.current_node.parent_node

    # @profile
    def choose_root_node(self, action):
        for child in self.root_node.child_list:
            if child.action == action:
                if not hasattr(child, 'legal_action_list'):
                    self.init_root_node()
                else:
                    self.root_node = child 
                    self.root_node.parent_node = None
                    self.root_node.action = None
                    self.root_node.prior_prob_P = None
                return
        print("ohohohoh")

    def init_root_node(self):
        with torch.no_grad():
            p, v = self.network([self.actor_state])
            p = self.softmax(p.squeeze()).cpu().numpy()
            v = v.item()
        self.root_node = Node(  
            player=(self.env.next_player+1)%2, 
            action=None, 
            prior_prob=None, 
            parent_node=None
        )
        self.root_node.choose_and_update(
            state=self.actor_state, 
            value=v,
            legal_action_list=list(np.asarray(range(self.env.action_space.n))[self.env.legal_action_mask]),
            legal_action_prior_list=p
        )

def play_with_me(board_size, win_size, network_fun, network_path, is_AI_first=False, *args, **kwargs):
    env = TicTacToeEnv(board_size=board_size, win_size=win_size) 
    state = env.reset()
    env.render()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    network = network_fun(env.observation_space.shape, env.action_space.n, *args, **kwargs).cuda()
    network.load_state_dict(torch.load(network_path))
    AI_player = 0 if is_AI_first else 1
    current_player = 0

    while not done:
        if current_player == AI_player:
            p, v = network([state])
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
        state, _, done, infos = env.step(action)
        env.render()
        print(infos)
        current_player = (current_player+1)%2
