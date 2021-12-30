
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
from utils.ActorAsync import NetworkActorAsync
from copy import deepcopy
import random
import numpy as np

class Nature_DQN:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        '''
        seed
        gamma
        clip_gradient
        eps_start
        eps_end
        eps_decay_steps
        ep_reward_avg_number
        sim_steps
        train_network_freq
        train_start_step
        train_update_target_freq
        eval_freq
        eval_number
        eval_max_steps
        eval_eps
        eval_video_fps
        eval_display
        '''
        self.args = args 
        self.kwargs = kwargs 
        self._init_seed()
        self.env = make_env_fun(*args, **kwargs)

        kwargs['policy_class'] = network_fun.__name__
        kwargs['env_name'] = self.env.unwrapped.spec.id
        logger.init(*args, **kwargs)

        if type(self) is Nature_DQN:
            self.network_lock = mp.Lock()
            self.train_actor = NetworkActorAsync(env = self.env, network_lock=self.network_lock, *args, **kwargs)
            self.train_actor.start()
            self.eval_actor = NetworkActorAsync(env = self.env, network_lock=mp.Lock(), *args, **kwargs)
            self.eval_actor.start()
            self.replay_buffer = ReplayBufferAsync(*args, **kwargs)
            self.replay_buffer.start()

            self.current_network = network_fun(self.env.observation_space.shape, self.env.action_space.n, *args, **kwargs).cuda().share_memory() 
            self.target_network  = network_fun(self.env.observation_space.shape, self.env.action_space.n, *args, **kwargs).cuda() 
            self.optimizer = optimizer_fun(self.current_network.parameters()) 
            self.update_target()

    def _init_seed(self):
        torch.manual_seed(self.kwargs['seed'])
        torch.cuda.manual_seed(self.kwargs['seed'])
        random.seed(self.kwargs['seed'])
        np.random.seed(self.kwargs['seed'])

    def update_target(self):
        self.target_network.load_state_dict(self.current_network.state_dict())
    
    def line_schedule(self, steps_idx):
        eps = self.kwargs['eps_end'] + (self.kwargs['eps_start'] - self.kwargs['eps_end']) * (1 - min(steps_idx,self.kwargs['eps_decay_steps']) / self.kwargs['eps_decay_steps'])
        return eps
    
    def train(self):
        last_sim_steps_idx, ep_idx = 1, 1
        ep_reward_list = deque(maxlen=self.kwargs['ep_reward_avg_number'])
        tic   = time.time()
        self.train_actor.update_policy(network = self.current_network)
        for sim_steps_idx in range(1, self.kwargs['sim_steps'] + 1, self.kwargs['train_network_freq']):
            eps = self.line_schedule(sim_steps_idx-self.kwargs['train_start_step']) if sim_steps_idx > self.kwargs['train_start_step'] else 1
            data = self.train_actor.collect(steps_number = self.kwargs['train_network_freq'], eps=eps)
            for frames_idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs, reward, done)
                if info is not None and info['episodic_return'] is not None:
                    episodic_steps = sim_steps_idx + frames_idx - last_sim_steps_idx
                    ep_reward_list.append(info['episodic_return'])
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    logger.add({'sim_steps':sim_steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': info['episodic_return'], 'ep_reward_avg': mean(ep_reward_list), 'eps': eps, 'fps': fps})
                    logger.wandb_print('(Training Agent) ', step=sim_steps_idx) if sim_steps_idx > self.kwargs['train_start_step'] else logger.wandb_print('(Collecting Data) ', step=sim_steps_idx)
                    ep_idx += 1
                    last_sim_steps_idx = sim_steps_idx + frames_idx

            if sim_steps_idx > self.kwargs['train_start_step']:
                self.compute_td_loss()

            if (sim_steps_idx-1) % self.kwargs['train_update_target_freq'] == 0:
                self.update_target()

            if (sim_steps_idx-1) % self.kwargs['eval_freq'] == 0:
                self.eval_actor.update_policy(network = deepcopy(self.current_network))
                self.eval_actor.eval(eval_idx = sim_steps_idx, eval_number = self.kwargs['eval_number'], eval_max_steps = self.kwargs['eval_max_steps'], eps = self.kwargs['eval_eps'])
                self.eval_actor.save_policy(name = sim_steps_idx)
                self.eval_actor.render(name=sim_steps_idx, render_max_steps=self.kwargs['eval_max_steps'], render_mode='rgb_array',fps=self.kwargs['eval_video_fps'], is_show=self.kwargs['eval_display'], eps = self.kwargs['eval_eps'])

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

# %%
