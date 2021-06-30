
#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
import random
import torch
import torch.nn as nn
import torch.optim as optim
from Network import *
from collections import deque
import argparse
from Config import get_default_parser
import wandb
import numpy as np
import time
from wrapper import make_env
from statistics import mean
from ReplayBufferAsync import ReplayBufferAsync
from LogAsync import logger
import torch.multiprocessing as mp
from ActorAsync import ActorAsync
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
import torch.multiprocessing as mp
from EvaluationAsync import EvaluationAsync

class DQN:
    def __init__(self, make_env_fun, netowrk_fun, optimizer_fun, *arg, **args):
        self.arg = arg 
        self.args = args 
        self.env = make_env_fun(args['env_name'], **args)
        self.batch_size=args['batch_size']
        self.gamma=args['gamma']
        self.gradient_clip = args['gradient_clip']
        self.eps_start = args['eps_start']
        self.eps_end = args['eps_end']
        self.eps_decay_steps = args['eps_decay_steps']
        self.train_steps = args['train_steps']
        self.start_training_steps = args['start_training_steps']
        self.update_target_steps = args['update_target_steps']
        self.train_freq = args['train_freq']
        self.seed = args['seed']

        
        self.eval_freq = args['eval_freq']

        self.lock = mp.Lock()
        self.actor = ActorAsync(self.env, steps_no=self.train_freq, seed=self.seed, lock=self.lock)
        self.replay_buffer = ReplayBufferAsync(args['buffer_size'], args['batch_size'], seed = self.seed, stack_frames=self.env.observation_space.shape[0])
        self.evaluator = EvaluationAsync(env = self.env, **args)

        self.current_network = netowrk_fun(self.env.observation_space.shape, self.env.action_space.n, **args).cuda().share_memory()
        self.target_network  = netowrk_fun(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        
        self.optimizer = optimizer_fun(self.current_network.parameters())
        self.update_target()
        self.evaluator.init(netowrk_fun)

    def update_target(self):
        self.target_network.load_state_dict(self.current_network.state_dict())
    
    def obtain_eps(self, steps_idx):
        eps = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(steps_idx,self.eps_decay_steps) / self.eps_decay_steps)
        return eps
    
    def train(self):
        last_steps_idx, ep_idx = 1, 1
        ep_reward_10_list = deque(maxlen=10)
        loss  = torch.tensor(0)
        tic   = time.time()
        self.actor.set_network(self.current_network)
        for self.train_steps_idx in range(1, self.train_steps + 1, self.train_freq):
            eps = self.obtain_eps(self.train_steps_idx-self.start_training_steps) if self.train_steps_idx > self.start_training_steps else 1
            data = self.actor.step(eps)
            for idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs[None,-1], reward, done)
                if info is not None and info['episodic_return'] is not None:
                    episodic_steps = self.train_steps_idx + idx - last_steps_idx
                    ep_reward_10_list.append(info['episodic_return'])
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    ep_reward_10_list_mean = mean(ep_reward_10_list)

                    logger.add({'train_steps':self.train_steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': info['episodic_return'], 'ep_reward_avg': ep_reward_10_list_mean, 'loss': loss.item(), 'eps': eps, 'fps': fps})
                    logger.wandb_print('(Training Agent) ', step=self.train_steps_idx) if self.train_steps_idx > self.start_training_steps else logger.wandb_print('(Collecting Data) ', step=self.train_steps_idx)
                    ep_idx += 1
                    last_steps_idx = self.train_steps_idx + idx

            if self.train_steps_idx > self.start_training_steps:
                loss = self.compute_td_loss()

            if self.train_steps_idx % self.update_target_steps == 1:
                self.update_target()
                
            if self.train_steps_idx % self.eval_freq == 1:
                self.evaluator.eval(train_steps=self.train_steps_idx, state_dict=self.current_network.state_dict())

class CatDQN(DQN):
    def __init__(self, make_env, network, optimizer, *arg, **args):
        super().__init__(make_env, network, optimizer, *arg, **args)
        self.num_atoms = args['num_atoms']
        self.v_min = args['v_min']
        self.v_max = args['v_max']
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        self.atoms_cpu = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.atoms_gpu = self.atoms_cpu.cuda()
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).cuda()
        self.torch_range = torch.arange(self.batch_size).long().cuda()

        
    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            prob_next = self.target_network(next_state)
            q_next = (prob_next * self.atoms_gpu).sum(-1)
            a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.torch_range, a_next, :]

        rewards = reward.unsqueeze(-1)
        atoms_target = rewards + self.gamma * (~done).unsqueeze(-1) * self.atoms_gpu.view(1, -1)
        atoms_target.clamp_(self.v_min, self.v_max).unsqueeze_(1)
        
        target_prob = (1 - (atoms_target - self.atoms_gpu.view(1, -1, 1)).abs() / self.delta_z).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.current_network.forward_log(state)
        log_prob = log_prob[self.torch_range, action, :]
        loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_network.parameters(), self.gradient_clip)
        gradient_norm = nn.utils.clip_grad_norm_(self.current_network.parameters(), self.gradient_clip)
        logger.add({'gradient_norm': gradient_norm.item()})
        with self.lock:
            self.optimizer.step()

        return loss

    

# %%
