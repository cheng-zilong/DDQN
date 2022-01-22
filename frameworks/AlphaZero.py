# 现在这个问题：样本产生速度太慢，batch size太小
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
from utils.ReplayBufferProcess import ReplayBufferProcess
from utils.LogProcess import logger
import torch.multiprocessing as mp
from utils.ActorProcess import NetworkActorAsync
from copy import deepcopy
import random
import numpy as np
from tqdm import tqdm
from utils.ActorProcess import BaseActorProcess
from numba import njit
from torch import optim
from torch.multiprocessing import Value
import os
import gym
from collections import namedtuple
import itertools
class AlphaZero:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.dummy_env = make_env_fun(*args, **kwargs)

        kwargs['policy_class'] = 'AlphaZero'
        kwargs['env_name'] = self.dummy_env.__class__.__name__
        kwargs['project_name'] = 'AlphaZero'
        logger.init(*args, **kwargs)

        self.network_lock = mp.Lock()
        self.replay_buffer = ReplayBufferProcess(*args, **kwargs)
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

        self.network = network_fun(self.dummy_env.observation_space.shape, self.dummy_env.action_space.n, *args, **kwargs).cuda().share_memory() 
        self.optimizer = optimizer_fun(self.network.parameters()) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.kwargs['lr_decay_step_size'], gamma=self.kwargs['lr_decay_gamma']) #TODO add to config

    def _init_seed(self):
        torch.manual_seed(self.kwargs['seed'])
        torch.cuda.manual_seed(self.kwargs['seed'])
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])

    def train(self):
        self.network.train()
        for i in range(self.actors_num):
            self.mcts_actors_list[i].collect(self.network)
            time.sleep(60)
        
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
            
            time.sleep(1)


class Node:
    def __init__(self,  current_player: int, action: int, prior_prob: float, parent_node: Node, worker_id:int):
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
        self.worker_id = worker_id

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
            parent_node=self,
            worker_id=self.worker_id
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

class AlphaZeroActorAsync(BaseActorProcess):
    TotalActorNumber=0
    game_counter = Value('i', 0)
    def __init__(self, make_env_fun, replay_buffer:ReplayBufferProcess, network_lock, workers_num = 64, *args, **kwargs):
        super().__init__(make_env_fun=make_env_fun, *args, **kwargs)
        self.envs_list = [make_env_fun(*args, **kwargs) for _ in range(workers_num)]
        self._network_lock = network_lock
        self.replay_buffer = replay_buffer
        self.simulator_env:TicTacToeEnv = deepcopy(self.env)
        self.actor_id = AlphaZeroActorAsync.TotalActorNumber
        AlphaZeroActorAsync.TotalActorNumber+=1

    def collect(self, network:AlphaZeroNetwork, *args, **kwargs):
        kwargs['network'] = network
        self.send(self.ASYNC_COLLECT, (args, kwargs))

    # @profile
    def _async_collect(self, network:AlphaZeroNetwork):
        _game_lines_dict = dict()
        GameData = namedtuple('GameData', 'player action state')
        # {0: (current_player, action, state]), ... 
        #  1: (current_player, action, state]), ... }
        self.shared_network = network 
        self.network = deepcopy(self.shared_network)
        self.network.eval()
        _info_dict = dict()
        while True:
            if self.actor_done_flag:
                try:
                    logger.add({
                        'game_count': AlphaZeroActorAsync.game_counter.value,
                        'mcts_time':mean(_mcts_time_list),
                        'game_time':time.time() - _game_tic,
                        'game_steps':_ep_step_idx/len(self.envs_list),
                        # 'replay_buffer_size':    str(self.replay_buffer.check_size()),
                    })
                except:
                    logger.add({
                        'game_count': 0,
                        'mcts_time':-1,
                        'game_time':-1,
                        'game_steps':-1,
                        # 'replay_buffer_size':    str(self.replay_buffer.check_size()),
                    })
                # logger.wandb_print('(Single Process Test)', 0)

                _ep_step_idx, _mcts_step_idx, _mcts_time_list, _game_tic = 0, 0, [], time.time()
                self.actor_done_flag = False 
                self.update_network()
                with AlphaZeroActorAsync.game_counter.get_lock(): AlphaZeroActorAsync.game_counter.value += len(self.envs_list)
                for woker_idx, game_line in _game_lines_dict.items():
                    for is_flip, rot in itertools.product([False, True],[0,1,2,3]):# 利用棋盘旋转对称性
                        for idx, (current_player, action_prob, state) in enumerate(game_line):
                            if action_prob is None:
                                flip_and_rotate_state = np.flip(np.rot90(state, k = rot, axes=(1,2)), 1) if is_flip else np.rot90(state, k = rot, axes=(1,2))
                                self.replay_buffer.add(
                                    action=None, 
                                    obs=flip_and_rotate_state, 
                                    reward=0, 
                                    done=False
                                )
                                continue
                            flip_and_rotate_action = np.rot90(action_prob.reshape(int(np.sqrt(len(action_prob))), -1), k = rot, axes=(0,1))
                            flip_and_rotate_state = np.rot90(state, k = rot, axes=(1,2))
                            if is_flip:
                                flip_and_rotate_action = np.flip(flip_and_rotate_action, 0)
                                flip_and_rotate_state = np.flip(flip_and_rotate_state, 1)
                            if _info_dict[woker_idx]['winner'] == -1 or action_prob is None:
                                reward = 0
                            elif _info_dict[woker_idx]['winner'] == current_player:
                                reward = 1
                            else:
                                reward = -1
                            self.replay_buffer.add(
                                action=flip_and_rotate_action.reshape(-1), 
                                obs=flip_and_rotate_state, 
                                reward=reward, 
                                done=False if idx!= len(game_line)-1 else True
                            )
                states_list = [env.reset() for env in self.envs_list]
                root_node_list = [self.init_root_node(env, self.network, idx) for idx, env in enumerate(self.envs_list)]
                for idx in range(len(self.envs_list)):
                    _game_lines_dict[idx] = [GameData(root_node_list[idx].current_player, None, states_list[idx])]
            _mcts_tic = time.time()
            actions, actions_prob = self.mcts(
                network=self.network, 
                temperature=0.1 if _mcts_step_idx >= 2 else 1, 
                root_node_list=root_node_list
            ) # TODO add 2 config
            _mcts_time_list.append(time.time()-_mcts_tic)
            done_list = [False] * len(root_node_list)
            for root_node_idx, (root_node, action) in enumerate(zip(root_node_list[:], actions[:])):
                _ep_step_idx+=1
                state, _, done_list[root_node_idx], _info_dict[root_node.worker_id] = self.envs_list[root_node.worker_id].step(action)
                _game_lines_dict[root_node.worker_id].append(GameData(
                    (self.envs_list[root_node.worker_id].next_player+1)%2, 
                    actions_prob[root_node_idx], 
                    state
                ))
            for root_node_idx, root_node in enumerate(root_node_list[:]):
                if done_list[root_node_idx]:
                    root_node_list.remove(root_node)
            actions = np.delete(actions, np.where(done_list))
            if len(root_node_list)==0:
                self.actor_done_flag = True
            self.update_root_node_list(
                root_node_list=root_node_list, 
                actions= actions, 
                envs=self.envs_list, 
                network=self.network
            )
            _mcts_step_idx+=1

    def update_network(self):
        with self._network_lock:
            self.network.load_state_dict(self.shared_network.state_dict())
    
    # @profile
    def mcts(self, network, temperature:int, root_node_list:list[Node]):
        # for i in tqdm(range(self.kwargs['mcts_sim_num'])):
        network.eval()
        for i in range(self.kwargs['mcts_sim_num']):
            _leaf_node_list = self._select(network, root_node_list, self.simulator_env) 
            # now the root_node_list is the list of leaf nodes
            self._expand_and_envaluate_(_leaf_node_list) 
            # now the root_node_list is the list of initialized leaf nodes
            _root_node_list = self._backup(_leaf_node_list)
             # now the root_node_list is the list of root nodes
        actions_prob = np.zeros((len(_root_node_list), self.env.action_space.n), dtype=np.float32)
        actions = np.zeros(len(_root_node_list), dtype=np.int32)
        for idx, root_node in enumerate(_root_node_list):
            _child_visit_num = np.asarray([child.visit_num_N for child in root_node.child_list])
            _temp_den = np.sum(_child_visit_num**(1/temperature))
            for child in root_node.child_list:
                actions_prob[idx, child.action] = (child.visit_num_N**(1/temperature))/_temp_den
            assert (np.sum(actions_prob[idx]) > 0.99 and np.sum(actions_prob[idx]) < 1.01), "Probability Error"
            actions[idx] = np.random.choice(np.arange(self.env.action_space.n), p = actions_prob[idx])
        return actions, actions_prob

    
    @staticmethod
    # @profile
    def _select(network:AlphaZeroNetwork, root_node_list:list[Node], simulator_env:gym.Env):
        current_node_list = root_node_list[:]
        _new_state_list = []
        _legal_action_mask_list = []
        _softmax = nn.Softmax(dim=1)
        for idx in range(len(current_node_list)):
            while len(current_node_list[idx].child_list) != 0:
                 # if len==0, then it is a leaf node or terminal node 
                temp_list = [puct(current_node_list[idx].visit_num_N, child.visit_num_N, child.prior_prob_P, child.mean_win_num_Q) 
                    for child in current_node_list[idx].child_list
                ]
                current_node_list[idx] = current_node_list[idx].child_list[np.argmax(temp_list)]
            # if this is the init state, it has been chosen and update
            if current_node_list[idx].is_terminal or current_node_list[idx].parent_node is None:
                continue
            # Now a leaf node is chosen, because the leaf node is initlized with only the player, action, and prior prob, generate its state firstly
            _new_state = simulator_env.reset_with_state_n_action(
                current_node_list[idx].parent_node.state, 
                current_node_list[idx].action
            )
            # if this is a terminal leaf, update immediately
            if simulator_env.status != 0: 
                current_node_list[idx].choose_and_update(
                    state=_new_state, 
                    value=0 if simulator_env.winner == -1 else 1, # draw=0, win=1
                    legal_action_list=None,
                    legal_action_prior_list=None,
                    is_terminal=True
                ) 
            # if this is not a terminal leaf, store and update it later
            else: 
                _new_state_list.append(_new_state)
                _legal_action_mask_list.append(simulator_env.legal_action_mask)
        if len(_new_state_list)==0:
            return current_node_list
        with torch.no_grad():
            p, v = network(_new_state_list)
            p.masked_fill_(mask = ~torch.tensor(_legal_action_mask_list, device='cuda:0'), value=-inf)# the illegal moves cannot be chosen
            p = _softmax(p).cpu().numpy()
            v = v.cpu().numpy().squeeze(-1)
        # Update the remaining leaf nodes
        _not_init_idx=0
        for current_node in current_node_list:
            if current_node.is_terminal or current_node.parent_node is None:
                continue
            current_node.choose_and_update(
                state=_new_state_list[_not_init_idx], 
                value=v[_not_init_idx],
                legal_action_list=list(np.arange(simulator_env.action_space.n)[_legal_action_mask_list[_not_init_idx]]),
                legal_action_prior_list=p[_not_init_idx],
                is_terminal=False
            ) 
            assert (np.sum(p[_not_init_idx])>0.99 and np.sum(p[_not_init_idx]) < 1.01), "Probability Error"
            _not_init_idx+=1
        assert _not_init_idx==len(_new_state_list), "Algorithm Error"
        return current_node_list

    
    @staticmethod
    # @profile
    def _expand_and_envaluate_(leaf_node_list:list[Node]):
        for leaf_node in leaf_node_list:
            if leaf_node.is_terminal:
                continue 
            if __debug__: _total_prob=0
            if leaf_node.parent_node is None: 
                # if this is the root node, add dirichlet noise
                dirichlet_array = np.random.dirichlet(np.ones(len(leaf_node.legal_action_list))*0.3).astype(np.float32) #TODO add 0.3 config
                for action_idx, action in enumerate(leaf_node.legal_action_list):
                    _prior_prob = 0.75 * leaf_node.legal_action_prior_list[action] + 0.25*dirichlet_array[action_idx]# TODO add 0.75 and 0.25 config
                    if __debug__: _total_prob += _prior_prob
                    leaf_node.generate_child(   
                        current_player=(leaf_node.current_player+1)%2, 
                        action=action,
                        prior_prob=_prior_prob
                    )
                assert (_total_prob>0.99 and _total_prob<1.01), "Probability Error"

            else:
                for action in leaf_node.legal_action_list:
                    if __debug__: _total_prob += leaf_node.legal_action_prior_list[action]
                    leaf_node.generate_child(   
                        current_player=(leaf_node.current_player+1)%2, 
                        action=action,
                        prior_prob=leaf_node.legal_action_prior_list[action]
                    )
                assert (_total_prob>0.99 and _total_prob<1.01), "Probability Error"
    # @profile
    @staticmethod
    def _backup(leaf_node_list:list[Node]):
        current_node_list:list[Node] = [None]*len(leaf_node_list)
        for idx, current_node in enumerate(leaf_node_list):
            leaf_node = current_node
            while current_node is not None:
                current_node.visit_num_N += 1
                if current_node.current_player == leaf_node.current_player:
                    current_node.win_num_W += leaf_node.value
                else:
                    current_node.win_num_W -= leaf_node.value
                current_node.mean_win_num_Q = current_node.win_num_W/current_node.visit_num_N
                if current_node.parent_node is None: 
                    # this is the root node
                    current_node_list[idx] = current_node
                    break
                current_node = current_node.parent_node
        return current_node_list
        
    # @profile
    @staticmethod
    def update_root_node_list(root_node_list:list[Node], actions:list[int], envs:list[gym.Env], network:nn.Module):
        for idx, root_node in enumerate(root_node_list[:]):
            for child in root_node.child_list:
                if child.action == actions[idx]:
                    if not hasattr(child, 'legal_action_list'):
                        root_node_list[idx] = AlphaZeroActorAsync.init_root_node(envs[root_node.worker_id], network, root_node.worker_id)
                    else:
                        root_node_list[idx] = child 
                        root_node_list[idx].parent_node = None
                        root_node_list[idx].action = None
                        root_node_list[idx].prior_prob_P = None
        return root_node_list

    @staticmethod
    def init_root_node(env:gym.Env, network:nn.Module, worker_id:int):
        assert env.status == 0, "Env Status Error" 
        with torch.no_grad():
            p, v = network([env.state])
            p:torch.Tensor
            p.masked_fill_(mask = ~torch.tensor(env.legal_action_mask, device='cuda:0'), value=-inf)
            p = nn.Softmax(dim=1)(p).cpu().numpy()
            v = v.item()
        root_node = Node(  
            current_player=(env.next_player+1)%2, 
            action=None, 
            prior_prob=None, 
            parent_node=None,
            worker_id = worker_id
        )
        root_node.choose_and_update(
            state=env.state, 
            value=v,
            legal_action_list=list(np.arange(env.action_space.n)[env.legal_action_mask]),
            legal_action_prior_list=p[0]
        )
        return root_node

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
        AI_player.update_root_node_list(action, env, network)
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