import torch
import torch.nn as nn
from utils.Network import *
from collections import deque
import time
from statistics import mean
from utils.ReplayBufferProcess import ReplayBufferProcess
from utils.LogProcess import logger
import torch.multiprocessing as mp
from utils.ActorProcess import DDPG_NetworkActorProcess
from copy import deepcopy
import random
import numpy as np
from frameworks.Vanilla_DDPG import Vanilla_DDPG_Async

class Vanilla_TD3_Async(Vanilla_DDPG_Async):
    def compute_td_loss(self):
        state, action, reward, next_state, done = self.process_dict['replay_buffer'].sample()
        with torch.no_grad():
            a_next = self.target_network.actor_forward(next_state).cpu()
            if not hasattr(self, 'train_idx'):
                self._action_space_low = torch.tensor([self.dummy_env.action_space.low]*a_next.shape[0])
                self._action_space_high = torch.tensor([self.dummy_env.action_space.high]*a_next.shape[0])
                self.train_idx = 1
                self._noise_mu = torch.zeros(a_next.shape)
                self._noise_sigma = torch.ones(a_next.shape)*self.kwargs['actor_noise']
            temp_noise = torch.clip(
                torch.normal(self._noise_mu, self._noise_sigma), 
                -self.kwargs['noise_clip'], 
                self.kwargs['noise_clip']
            ) 
            a_next = torch.clip(
                a_next+temp_noise, 
                self._action_space_low, 
                self._action_space_high
            ).cuda()
            q_nexts = self.target_network.critic_forward(next_state, a_next)
            q_targets = torch.zeros(q_nexts.shape).cuda()
            for idx in range(q_nexts.shape[1]):
                q_targets[:, idx] = reward + self.kwargs['gamma'] * (~done) * q_nexts[:, idx]
            q_targets = torch.min(q_targets, dim=1)[0].view(-1,1)
        
        # train critic
        q = self.current_network.critic_forward(state, action)
        loss_critic = nn.MSELoss()(q_targets.broadcast_to(q.shape), q)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        with self.network_lock:
            self.optimizer_critic.step()
        logger.add({'loss_critic': loss_critic.item()})
        # train actor
        if (self.train_idx-1) % self.kwargs['train_actor_freq'] == 0:
            predict_action = self.current_network.actor_forward(state)
            loss_actor = -torch.mean(self.current_network.critic_forward(state, predict_action))
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            with self.network_lock:
                self.optimizer_actor.step()
            logger.add({'loss_actor': loss_actor.item()})
        self.train_idx+=1