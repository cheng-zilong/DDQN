
#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
import torch
import torch.nn as nn
from utils.Network import *
from collections import deque
import time
from statistics import mean
from utils.ReplayBufferAsync import ReplayBufferAsync
from utils.LogAsync import logger
import torch.multiprocessing as mp
from utils.ActorAsync import ActorAsync
import torch.multiprocessing as mp
from utils.EvaluatorAsync import EvaluatorAsync
from copy import deepcopy
import random
import numpy as np

class MyActorAsync(ActorAsync):
    def __init__(self, env, network_lock, seed = None, *arg, **args):
        super().__init__(env, seed, *arg, **args)
        self._network_lock = network_lock

    def _step(self, eps, *arg, **args):
        '''
        epsilon greedy step
        '''
        # auto reset
        if not hasattr(self,'_actor_done_flag') or self._actor_done_flag:
            self._actor_last_state = self.env.reset()
            self._actor_done_flag = False
            return [None, self._actor_last_state, None, None, None]
        eps_prob =  random.random()
        if eps_prob > eps:
            with self._network_lock:
                action = self._act_network.act(np.array(self._actor_last_state, copy=False))
        else:
            action = self.env.action_space.sample()
        self._actor_last_state, reward, self._actor_done_flag, info = self.env.step(action)
        return [action, self._actor_last_state, reward, self._actor_done_flag, info]

    def _update_policy(self, network = None, *arg, **args):
        if network is not None:
            self._act_network = network

    def _save_policy(self, *arg, **args):
        '''
        This is only a template for the _save_policy method
        Overwrite this method to implement your own _save_policy method
        '''
        pass

class Nature_DQN:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *arg, **args):
        self.arg = arg 
        self.args = args 
        env = make_env_fun(**args)
        self.gamma=args['gamma']
        self.gradient_clip = args['gradient_clip']
        self.eps_start = args['eps_start']
        self.eps_end = args['eps_end']
        self.eps_decay_steps = args['eps_decay_steps']
        self.start_training_steps = args['start_training_steps']
        self.update_target_freq = args['update_target_freq']
        self.eval_freq = args['eval_freq']

        if not hasattr(self, 'actor'):
            self._network_lock = mp.Lock()
            self.train_actor = MyActorAsync(env = env, network_lock=self._network_lock, *arg, **args)
            self.train_actor.start()
        if not hasattr(self, 'evaluator'):
            self.eval_actor = MyActorAsync(env = env, network_lock=mp.Lock(), *arg, **args)
            self.eval_actor.start()
        self.replay_buffer = ReplayBufferAsync(*arg, **args)
        self.replay_buffer.start()

        self.current_network = network_fun(env.observation_space.shape, env.action_space.n, **args).cuda().share_memory()
        self.target_network  = network_fun(env.observation_space.shape, env.action_space.n, **args).cuda()
        self.evaluator_network = network_fun(env.observation_space.shape, env.action_space.n, **args).cuda().share_memory()

        self.optimizer = optimizer_fun(self.current_network.parameters())
        self.update_target()
        
    def update_target(self):
        self.target_network.load_state_dict(self.current_network.state_dict())
    
    def line_schedule(self, steps_idx):
        eps = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(steps_idx,self.eps_decay_steps) / self.eps_decay_steps)
        return eps
    
    def train(self):
        last_train_steps_idx, ep_idx = 1, 1
        ep_reward_list = deque(maxlen=self.args['ep_reward_avg_number'])
        loss  = torch.tensor(0)
        tic   = time.time()
        self.train_actor.update_policy(network = self.current_network)
        for train_steps_idx in range(1, self.args['train_steps'] + 1, self.args['train_freq']):
            eps = self.line_schedule(train_steps_idx-self.start_training_steps) if train_steps_idx > self.start_training_steps else 1
            data = self.train_actor.collect(steps_no = self.args['train_freq'], eps=eps)
            for frames_idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs, reward, done)
                if info is not None and info['episodic_return'] is not None:
                    episodic_steps = train_steps_idx + frames_idx - last_train_steps_idx
                    ep_reward_list.append(info['episodic_return'])
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    logger.add({'train_steps':train_steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': info['episodic_return'], 'ep_reward_avg': mean(ep_reward_list), 'loss': loss.item(), 'eps': eps, 'fps': fps})
                    logger.wandb_print('(Training Agent) ', step=train_steps_idx) if train_steps_idx > self.start_training_steps else logger.wandb_print('(Collecting Data) ', step=train_steps_idx)
                    ep_idx += 1
                    last_train_steps_idx = train_steps_idx + frames_idx

            if train_steps_idx > self.start_training_steps:
                loss = self.compute_td_loss()

            if (train_steps_idx-1) % self.update_target_freq == 0:
                self.update_target()

            if (train_steps_idx-1) % self.eval_freq == 0:
                self.evaluator_network.load_state_dict(self.current_network.state_dict())
                self.eval_actor.update_policy(network = self.evaluator_network)
                self.eval_actor.eval(eval_idx = train_steps_idx, eps = self.args['eval_eps'])
                # self.evaluator.save_policy(network = self.current_network) TODO

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            q_next = self.target_network(next_state).max(1)[0]
            q_target = reward + self.gamma * (~done) * q_next 
        
        q = self.current_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = nn.MSELoss()(q_target, q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_network.parameters(), self.gradient_clip)
        gradient_norm = nn.utils.clip_grad_norm_(self.current_network.parameters(), self.gradient_clip)
        logger.add({'gradient_norm': gradient_norm.item()})
        with self._network_lock:
            self.optimizer.step()
        return loss

# %%
