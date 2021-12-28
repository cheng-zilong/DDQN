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
from utils.ActorAsync import MultiPlayerSequentialGameNetworkActorAsync 
from copy import deepcopy
import random
import numpy as np

class Nature_DQN_TwoPlayer:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        '''
        seed
        gamma
        clip_gradient
        eps_start
        eps_end
        eps_decay_steps
        train_start_step
        train_update_target_freq
        eval_freq
        '''
        self.arg = args 
        self.args = kwargs 
        self._init_seed()
        self.player_number = 2

        self.env = make_env_fun(**kwargs)
        self.network_lock = mp.Lock()
        self.train_actor = MultiPlayerSequentialGameNetworkActorAsync(env = self.env, network_lock=self.network_lock, *args, **kwargs)
        self.train_actor.start()
        self.eval_actor = MultiPlayerSequentialGameNetworkActorAsync(env = self.env, network_lock=mp.Lock(), *args, **kwargs)
        self.eval_actor.start()
    
        self.replay_buffer_list = []
        for _ in range(self.player_number):
            self.replay_buffer_list.append(ReplayBufferAsync(*args, **kwargs))
            self.replay_buffer_list[-1].start()

        self.current_network_list, self.target_network_list, self.optimizer_list = [], [], []
        for _ in range(self.player_number):
            self.current_network_list.append(network_fun(self.env.observation_space.shape, self.env.action_space.n, **kwargs).cuda().share_memory())
            self.target_network_list.append(network_fun(self.env.observation_space.shape, self.env.action_space.n, **kwargs).cuda() )
            self.optimizer_list.append(optimizer_fun(self.current_network_list[-1].parameters()))
        self.update_target()

    def _init_seed(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])

    def update_target(self):
        for current_network, target_network in zip(self.current_network_list, self.target_network_list):
            target_network.load_state_dict(current_network.state_dict())
    
    def line_schedule(self, steps_idx):
        eps = self.args['eps_end'] \
            + (self.args['eps_start'] - self.args['eps_end']) * (1 - min(steps_idx,self.args['eps_decay_steps']) / self.args['eps_decay_steps'])
        return eps
    
    def train(self):
        last_sim_steps_idx, ep_idx, last_ep_idx = 0, 1, 0
        ep_reward_list = deque(maxlen=self.args['ep_reward_avg_number'])
        tic   = time.time()
        self.train_actor.update_policy(network_list = self.current_network_list)
        for sim_steps_idx in range(1, self.args['sim_steps'] + 1, self.args['train_network_freq']):
            eps = self.line_schedule(sim_steps_idx-self.args['train_start_step']) if sim_steps_idx > self.args['train_start_step'] else 1
            data = self.train_actor.collect(steps_number = self.args['train_network_freq'], eps=eps)
            for frames_idx, ep_data in enumerate(data):
                for player_idx, (action, obs, reward, done, info) in enumerate(ep_data):
                    self.replay_buffer_list[player_idx].add(action, obs, reward, done)
                    if info is not None and info['episodic_return'] is not None:
                        ep_reward_list.append(info['episodic_return'])
                        if ep_idx % self.args['train_log_freq'] == 0:
                            episodic_steps = sim_steps_idx + frames_idx*self.player_number + player_idx - last_sim_steps_idx
                            ep_numbers = ep_idx - last_ep_idx
                            toc = time.time()
                            fps = episodic_steps / (toc-tic) #TODO not correct
                            tic = time.time()
                            logger.add({'sim_steps':sim_steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps/ep_numbers, 'ep_reward': info['episodic_return'], 'ep_reward_avg': np.mean(np.array(ep_reward_list),axis=0), 'ep_reward_avg_player0': np.mean(np.array(ep_reward_list)[:,0]), 'eps': eps, 'fps': fps})
                            logger.wandb_print('(Training Agent) ', step=sim_steps_idx) if sim_steps_idx > self.args['train_start_step'] else logger.wandb_print('(Collecting Data) ', step=sim_steps_idx)
                            last_ep_idx = ep_idx
                            last_sim_steps_idx = sim_steps_idx + frames_idx*self.player_number + player_idx
                        ep_idx += 1
                        
            if sim_steps_idx > self.args['train_start_step']:
                self.compute_td_loss()

            if (sim_steps_idx-1) % self.args['train_update_target_freq'] == 0:
                self.update_target()

            if (sim_steps_idx-1) % self.args['eval_freq'] == 0:
                self.eval_actor.update_policy(network_list = deepcopy(self.current_network_list))
                # self.eval_actor.eval(eval_idx = sim_steps_idx, eval_number = self.args['eval_number'], eval_max_steps = self.args['eval_max_steps'], eps = self.args['eval_eps'])
                self.eval_actor.save_policy(name = sim_steps_idx)
                # self.eval_actor.render(name=sim_steps_idx, render_max_steps=self.args['eval_max_steps'], render_mode='rgb_array',fps=self.args['eval_video_fps'], is_show=self.args['eval_display'], eps = self.args['eval_eps'])

    def compute_td_loss(self):
        '''
        TODO
        Agent 网络训练可以并行
        '''
        gradient_norm_list = []
        loss_list = []
        for player_idx in range(self.player_number):
            state, action, reward, next_state, done = self.replay_buffer_list[player_idx].sample()
            with torch.no_grad():
                q_next = self.target_network_list[player_idx](next_state).max(1)[0]
                q_target = reward + self.args['gamma'] * (~done) * q_next 
            
            q = self.current_network_list[player_idx](state).gather(1, action.unsqueeze(-1)).squeeze(-1)
            loss = nn.MSELoss()(q_target, q)
            self.optimizer_list[player_idx].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.current_network_list[player_idx].parameters(), self.args['clip_gradient'])
            gradient_norm = nn.utils.clip_grad_norm_(self.current_network_list[player_idx].parameters(), self.args['clip_gradient'])
            gradient_norm_list.append(gradient_norm.item())
            loss_list.append(loss.item())
            with self.network_lock:
                self.optimizer_list[player_idx].step()
        logger.add({'gradient_norm': mean(gradient_norm_list) , 'loss': mean(loss_list)})
# %%
