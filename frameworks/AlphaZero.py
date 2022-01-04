from __future__ import annotations
import torch
import torch.nn as nn
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

class Node:
    CONST_C_BASE = 19652
    CONST_C_INIT = 2.5

    def __init__(self, player: int, action: int, state: np.array, prior_prob: float, parent_node: Node):
        self.player = player
        self.action = action
        self.state = state
        self.parent_node = parent_node
        self.child_list = []

        self.visit_num_N = 0 
        self.win_num_W = 0
        self.mean_win_num_Q = 0
        self.prior_prob_P = prior_prob
        
        self.is_terminal = False

    @property
    def puct(self):
        weighting_C_s = np.log((1+self.parent_node.visit_num_N+Node.CONST_C_BASE)/Node.CONST_C_BASE) + Node.CONST_C_INIT
        U_s_a = weighting_C_s*self.prior_prob_P*np.sqrt(self.parent_node.visit_num_N)/(1+self.visit_num_N)
        return self.mean_win_num_Q + U_s_a

    def generate_child(self, player: int, action: int, state: np.array, prior_prob: float):
        new_node = Node(player=player, action = action, state=state, prior_prob = prior_prob, parent_node=self)
        self.child_list.append(new_node)
        return new_node

class AlphaZero:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.env = make_env_fun(*args, **kwargs)

        self.current_network = network_fun(self.env.observation_space.shape, self.env.action_space.n, *args, **kwargs).cuda().share_memory() 
        self.optimizer = optimizer_fun(self.current_network.parameters()) 
        # kwargs['policy_class'] = 'AlphaZero'
        # kwargs['env_name'] = self.env.unwrapped.spec.id
        # logger.init(*args, **kwargs)

    def _init_seed(self):
        torch.manual_seed(self.kwargs['seed'])
        torch.cuda.manual_seed(self.kwargs['seed'])
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])
    
    def select(self):
        pass

    def expand_n_envaluate(self):
        pass

    def backup(self):
        pass


    def train(self):
        self.current_network.eval()
        state = self.env.reset()
        softmax = nn.Softmax(dim=1)
        for _ in tqdm(range(8000)):
            with torch.no_grad():
                p, v = self.current_network([state])
                p = softmax(p)
                p.squeeze()[~self.env.legal_action_mask] = -np.inf # remove illegal actions
                action = torch.argmax(p)
            state, _, done, info = self.env.step(action)
            if done:
                state = self.env.reset()
                tqdm.write(str(info))

            # loss = (v-0.5)**2 + nn.CrossEntropyLoss()(p, torch.tensor([10],device='cuda:0'))
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
        print("!!!!")

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            q_next = self.target_network(next_state).max(1)[0]
            q_target = reward + self.kwargs['gamma'] * (~done) * q_next 
        
        q = self.current_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = nn.MSELoss()(q_target, q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_network.parameters(), self.kwargs['clip_gradient'])
        gradient_norm = nn.utils.clip_grad_norm_(self.current_network.parameters(), self.kwargs['clip_gradient'])
        logger.add({'gradient_norm': gradient_norm.item(), 'loss': loss.item()})
        with self.network_lock:
            self.optimizer.step()
        return loss

