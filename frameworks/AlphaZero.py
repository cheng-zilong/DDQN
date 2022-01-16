from __future__ import annotations
from logging import root
from multiprocessing import Value
from matplotlib.pyplot import tick_params
from numpy.core.fromnumeric import argmax
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
import gym
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

            if train_idx % 10000 == 0: # TODO add config
                if not os.path.exists('save_model/' + logger._run_name + '/'):
                    os.makedirs('save_model/' + logger._run_name + '/')
                torch.save(self.network.state_dict(), 'save_model/' + logger._run_name + '/' + str(train_idx) +'.pt')

            time.sleep(2)

class Node:
    def __init__(self,  current_player: int, action: int, prior_prob: float, parent_node: Node):
        # current_player执行了action，得到了state
        self.current_player = current_player
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
            is_terminal=False
        ):
        self.state = state
        self.value = value # if this node is a leaf node, then the value is 
        self.legal_action_list = legal_action_list
        self.legal_action_prior_list = legal_action_prior_list
        self.is_terminal = is_terminal

    def generate_child(self,    
            current_player: int, 
            action: int, 
            prior_prob: float
        ):
        new_node = Node(    
            current_player=current_player, 
            action=action, 
            prior_prob=prior_prob, 
            parent_node=self
        )
        self.child_list.append(new_node)
        return new_node

    def plot_tree(self):
        f = open("tree.txt", "w")
        self._plot_tree(i=0, f=f)
        f.close

    def _plot_tree(self, i, f):
        f.write(' ' + ('-'*i) + str(self.action) + ' \t N:%5d \t W:%.5f \t Q:%.5f \t P:%.5f \t'%(self.visit_num_N, self.win_num_W,self.mean_win_num_Q, self.prior_prob_P if self.prior_prob_P is not None else -1) +
                f'is_termial:{str(self.is_terminal)}' + ((' \t value: %.5f'%self.value) if hasattr(self, 'value') else ('\t value: None')))
        f.write('\n')
        if hasattr(self, "state"):
            f.write(str(self.state))
            f.write('\n')
        for child in self.child_list:
            child._plot_tree(i=i+1, f=f)

@njit
def puct(p_visit_num:int, visit_num:int, prior_p:float, mean_win_num_Q:float) -> float:
    CONST_C_BASE = 19652 
    CONST_C_INIT = 1.5
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
        self.softmax = nn.Softmax(dim=0)

    def collect(self, network:AlphaZeroNetwork, *args, **kwargs):
        kwargs['network'] = network
        self.send(self.COLLECT, (args, kwargs))

    def _collect(self, network:AlphaZeroNetwork):
        game_line = []
        #[(current_player, action, state]), ... ]
        self.shared_network = network 
        self.network = deepcopy(self.shared_network)
        self.network.eval()
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
                for is_flip in [False, True]:# 利用棋盘旋转对称性
                    for rot in [0,1,2,3]:
                        for idx, (current_player, action_prob, state) in enumerate(game_line):
                            flip_and_rotate_action = action_prob.reshape(int(np.sqrt(len(action_prob))), -1) if action_prob is not None else None
                            flip_and_rotate_action = np.rot90(flip_and_rotate_action, k = rot, axes=(0,1)) if action_prob is not None else None
                            flip_and_rotate_state = np.rot90(state, k = rot, axes=(1,2))
                            if is_flip:
                                flip_and_rotate_action = np.flip(flip_and_rotate_action, 0) if action_prob is not None else None
                                flip_and_rotate_state = np.flip(flip_and_rotate_state, 1)
                            if info['winner'] == -1 or action_prob is None:
                                reward = 0
                            elif info['winner'] == current_player:
                                reward = 1
                            else:
                                reward = -1
                            self.replay_buffer.add(flip_and_rotate_action.reshape(-1) if action_prob is not None else None, flip_and_rotate_state, reward, False if idx!=game_line_size-1 else True)
                self.actor_state = self.env.reset()
                self.init_root_node(self.env, self.network)
                self.actor_done_flag = False 
                game_line = [(self.root_node.current_player, None, self.actor_state)]
                mcts_time_list = []
            mcts_tic = time.time()
            action, action_prob = self.mcts(self.network, 0 if self.epi_step_idx >= 2 else 1) # TODO add 2 config
            mcts_time_list.append(time.time()-mcts_tic)
            self.actor_state, _, self.actor_done_flag, info = self.env.step(action)
            game_line.append(((self.env.next_player+1)%2, action_prob, self.actor_state)) # next_player convert to current player
            self.change_root_node(action, self.env, self.network)
            self.epi_step_idx+=1

    def update_network(self):
        with self._network_lock:
            self.network.load_state_dict(self.shared_network.state_dict())
    
    # @profile
    def mcts(self, network, temperature):
        # for i in tqdm(range(self.kwargs['mcts_sim_num'])):
        network.eval()
        for i in range(self.kwargs['mcts_sim_num']):
            leaf_node = self._select(network = network, root_node = self.root_node)
            self._expand_and_envaluate(leaf_node)
            self._backup(leaf_node)
        # logger.terminal_print(caption = '------------(Actor Thread %d)'%self.actor_id, 
        #         log_dict_tmp={'------------ep_step': str(self.epi_step_idx)}
        #     )

        action_prob = np.zeros(self.simulator_env.action_space.n, dtype=np.float32)
        if temperature == 1: #
            for child in self.root_node.child_list:
                action_prob[child.action] = child.visit_num_N/(self.root_node.visit_num_N-1) # because the root node is chosen one more time
        elif temperature == 0:
            visit_list = [child.visit_num_N for child in self.root_node.child_list]
            action_prob[self.root_node.child_list[np.argmax(visit_list)].action] = 1
        else:
            raise Exception('temperature must be 0 or 1')
        action = np.random.choice(list(range(self.simulator_env.action_space.n)), p = action_prob)
        return action, action_prob

    # @profile
    def _select(self, network, root_node:Node):
        current_node = root_node
        while len(current_node.child_list) != 0: # Not a leaf node 
            temp_list = [puct(child.parent_node.visit_num_N, child.visit_num_N, child.prior_prob_P, child.mean_win_num_Q) for child in current_node.child_list]
            current_node = current_node.child_list[np.argmax(temp_list)]
        # if this is the init state, it has been chosen and update
        if current_node.is_terminal or current_node.parent_node is None:
            return current_node
        
        # Now a leaf node is chosen, because the leaf node is initlized with only the player, action, and prior prob, generate its state firstly
        new_state = self.simulator_env.reset_with_state_n_action(current_node.parent_node.state, current_node.action)
        
        if self.simulator_env.status != 0: #if this is the terminal leaf
            if self.simulator_env.winner == -1: # draw
                value = 0
            else: # if current player wins
                value = 1
            current_node.choose_and_update(
                state=new_state, 
                value=value,
                legal_action_list=None,
                legal_action_prior_list=None,
                is_terminal=True
            ) 
        else:
            # Evaluate the leaf node
            with torch.no_grad():
                p, v = network([new_state])
                p = p.view(-1)
                p[~self.simulator_env.legal_action_mask] = -inf # the illegal moves cannot be chosen
                p = self.softmax(p).cpu().numpy()
                v = v.item()
            # Update the leaf node
            legal_action_list = list(np.asarray(range(self.simulator_env.action_space.n))[self.simulator_env.legal_action_mask])
            current_node.choose_and_update(
                state=new_state, 
                value=v,
                legal_action_list=legal_action_list,
                legal_action_prior_list=p,
                is_terminal=False
            ) 
        return current_node

    # @profile
    def _expand_and_envaluate(self, leaf_node:Node):
        if leaf_node.is_terminal:
            return 
        if leaf_node.parent_node is None: # if this is the root node, add dirichlet noise
            dirichlet_array = np.random.dirichlet(np.ones(len(leaf_node.legal_action_list))*0.3).astype(np.float32) #TODO add 0.3 config
            for action_idx, action in enumerate(leaf_node.legal_action_list):
                leaf_node.generate_child(   
                    current_player=(leaf_node.current_player+1)%2, 
                    action=action,
                    prior_prob=0.75 * leaf_node.legal_action_prior_list[action] + 0.25*dirichlet_array[action_idx]# TODO add 0.75 and 0.25 config
                )
        else:
            for action in leaf_node.legal_action_list:
                leaf_node.generate_child(   
                    current_player=(leaf_node.current_player+1)%2, 
                    action=action,
                    prior_prob=leaf_node.legal_action_prior_list[action]
                )

    # @profile
    def _backup(self, leaf_node:Node):
        current_node = leaf_node
        while current_node is not None:
            current_node.visit_num_N += 1
            if current_node.current_player == leaf_node.current_player:
                current_node.win_num_W += leaf_node.value
            else:
                current_node.win_num_W -= leaf_node.value
            current_node.mean_win_num_Q = current_node.win_num_W/current_node.visit_num_N
            current_node = current_node.parent_node

    # @profile
    def change_root_node(self, action:int, env:gym.Env, network:nn.Module):
        for child in self.root_node.child_list:
            if child.action == action:
                if not hasattr(child, 'legal_action_list'):
                    self.init_root_node(env, network)
                else:
                    self.root_node = child 
                    self.root_node.parent_node = None
                    self.root_node.action = None
                    self.root_node.prior_prob_P = None
                return
        self.init_root_node(env, network)

    def init_root_node(self, env:gym.Env, network:nn.Module):
        with torch.no_grad():
            p, v = network([env.state])
            p = self.softmax(p.squeeze()).cpu().numpy()
            v = v.item()
        self.root_node = Node(  
            current_player=(env.next_player+1)%2, 
            action=None, 
            prior_prob=None, 
            parent_node=None
        )
        self.root_node.choose_and_update(
            state=env.state, 
            value=v,
            legal_action_list=list(np.asarray(range(env.action_space.n))[env.legal_action_mask]),
            legal_action_prior_list=p
        )

def play_with_me(make_env_fun, network_fun, network_path, is_AI_first=False, *args, **kwargs):
    env = make_env_fun(*args, **kwargs) 
    state = env.reset()
    env.render()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    network = network_fun(env.observation_space.shape, env.action_space.n, *args, **kwargs).cuda()
    network.load_state_dict(torch.load(network_path))
    AI_player = AlphaZeroActorAsync(make_env_fun, None, None, *args, **kwargs )
    AI_player.init_root_node(env, network)
    AI_player_idx = 0 if is_AI_first else 1
    current_player = 0
    while not done:
        p, v = network([state])
        print(p.reshape(int(np.sqrt(env.action_space.n)), -1))
        print(v)
        if current_player == AI_player_idx:
            # action = torch.argmax(p).item()
            action, _ = AI_player.mcts(network, 0)
            print(f'AI turn:{action}')
        else:
            while True:
                try:
                    x, y = input("Your turn: ").split()
                    break
                except ValueError:
                    print('Illegal Input! Please indicate "row" and "column".')
            action = (int(x))*kwargs['board_size']+(int(y))
        state, _, done, infos = env.step(action)
        AI_player.change_root_node(action, env, network)
        env.render()
        print(infos)
        current_player = (current_player+1)%2

def AI_play_again_AI(make_env_fun, network_fun, network_path, *args, **kwargs):
    env = make_env_fun(*args, **kwargs) 
    state = env.reset()
    env.render()
    done = False
    kwargs['policy_class'] = 'Vanilla_MCTS'
    kwargs['env_name'] = env.__class__.__name__
    # logger.init(*args, **kwargs)
    network = network_fun(env.observation_space.shape, env.action_space.n, *args, **kwargs).cuda()
    network.load_state_dict(torch.load(network_path))
    # AI_player = AlphaZeroActorAsync(make_env_fun, None, None, *args, **kwargs )
    # AI_player.init_root_node(env, network)
    current_player = 0
    while not done:
        p, v = network([state])
        print(p.reshape(int(np.sqrt(env.action_space.n)), -1))
        print(v)
        action = torch.argmax(p).item()
        # action, _ = AI_player.mcts(network, 0)
        print(f'AI turn:{action}')
        state, _, done, infos = env.step(action)
        # AI_player.change_root_node(action, env, network)
        env.render()
        print(infos)
        current_player = (current_player+1)%2