import torch
import torch.nn as nn
from utils.Network import *
from collections import deque
import time
from statistics import mean
from utils.LogProcess import logger
import torch.multiprocessing as mp
from utils.ActorProcess import DDPG_NetworkActorProcess
from copy import deepcopy
import random
import numpy as np
from frameworks.Vanilla_DQN import Vanilla_DQN_Async

class Vanilla_DDPG_Async(Vanilla_DQN_Async):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs):
        super().__init__(make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs)
        self.process_dict['train_actor'] = [
            DDPG_NetworkActorProcess(
                make_env_fun = self.make_env_fun, 
                replay_buffer=self.process_dict['replay_buffer'], 
                network_lock=self.network_lock, 
                *self.args, **self.kwargs
            ) for _ in range(actor_num)
        ]
        self.process_dict['eval_actor'] = DDPG_NetworkActorProcess(
            make_env_fun = self.make_env_fun, 
            network_lock=mp.Lock(), 
            *self.args, **self.kwargs
        )

    def init_network(self):
        self.current_network = self.network_fun(self.dummy_env.observation_space.shape, self.dummy_env.action_space.shape[0], *self.args, **self.kwargs).cuda().share_memory() 
        self.target_network  = self.network_fun(self.dummy_env.observation_space.shape, self.dummy_env.action_space.shape[0], *self.args, **self.kwargs).cuda() 
        self.optimizer_actor = self.optimizer_fun[0](self.current_network.policy_fc.parameters()) 
        self.optimizer_critic = self.optimizer_fun[1](self.current_network.value_fc.parameters()) 
        self.update_target(tau=1)

    def sigma_line_schedule(self, steps_idx):
        sigma = self.kwargs['sigma_end'] + (self.kwargs['sigma_start'] - self.kwargs['sigma_end']) * (1 - min(steps_idx,self.kwargs['sigma_decay_steps']) / self.kwargs['sigma_decay_steps'])
        return sigma

    def train(self):
        self.start_process()
        self.init_network()

        for actor in self.process_dict['train_actor']:
            actor.update_policy(network = self.current_network)

        while self.process_dict['replay_buffer'].check_size() < self.kwargs['train_start_step']:
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number=self.kwargs['train_network_freq'], sigma=1)

        for train_idx in range(1, self.kwargs['train_steps'] + 1):
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number=self.kwargs['train_network_freq'], sigma=self.sigma_line_schedule(train_idx), train_idx=train_idx)
            
            self.compute_td_loss()
            self.update_target(tau=self.kwargs['train_update_tau'])

            if (train_idx-1) % self.kwargs['eval_freq'] == 0:
                self.process_dict['eval_actor'].update_policy(network = deepcopy(self.current_network))
                self.process_dict['eval_actor'].eval(eval_idx = train_idx, eval_number = self.kwargs['eval_number'], eval_max_steps = self.kwargs['eval_max_steps'], sigma = self.kwargs['eval_sigma'])
                self.process_dict['eval_actor'].save_policy(name = train_idx)
                self.process_dict['eval_actor'].render(name=train_idx, render_max_steps=self.kwargs['eval_max_steps'], render_mode='rgb_array',fps=self.kwargs['eval_video_fps'], is_show=self.kwargs['eval_display'], sigma = self.kwargs['eval_sigma'])

    def update_target(self, tau):
        for target_param, current_param in zip(self.target_network.parameters(),self.current_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + current_param.data * tau
            )

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.process_dict['replay_buffer'].sample()
        with torch.no_grad():
            a_next = self.target_network.actor_forward(next_state)
            q_next = self.target_network.critic_forward(next_state, a_next)
            q_target = reward + self.kwargs['gamma'] * (~done) * q_next
        q = self.current_network.critic_forward(state, action)
        self.optimizer_critic.zero_grad()
        loss_critic = nn.MSELoss()(q_target, q)
        loss_critic.backward()
        with self.network_lock:
            self.optimizer_critic.step()

        predict_action = self.current_network.actor_forward(state)
        self.optimizer_actor.zero_grad()
        loss_actor = -torch.mean(self.current_network.critic_forward(state, predict_action))
        loss_actor.backward()
        with self.network_lock:
            self.optimizer_actor.step()
        logger.add({'loss_critic': loss_critic.item(), 'loss_actor': loss_actor.item()})