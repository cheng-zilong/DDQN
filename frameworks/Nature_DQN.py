
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
from utils.ReplayBufferProcess import ReplayBufferProcess
from utils.LogProcess import logger
import torch.multiprocessing as mp
from utils.ActorProcess import NetworkActorProcess
from copy import deepcopy
import random
import numpy as np

class Nature_DQN_Sync:
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *args, **kwargs):
        '''
        '''
        self.args = args 
        self.kwargs = kwargs 
        self.make_env_fun = make_env_fun
        self.network_fun = network_fun
        self.optimizer_fun = optimizer_fun
        self.dummy_env = self.make_env_fun(*self.args, **self.kwargs)
        self.kwargs['policy_class'] = network_fun.__name__
        self.kwargs['env_name'] = self.dummy_env.unwrapped.spec.id
        logger.init(*self.args, **self.kwargs)

        self.network_lock = mp.Lock()
        self.process_dict = {
            'replay_buffer': ReplayBufferProcess(*self.args, **self.kwargs),
            'train_actor': NetworkActorProcess(
                make_env_fun = self.make_env_fun, 
                network_lock=self.network_lock, 
                *self.args, **self.kwargs
            ),
            'eval_actor': NetworkActorProcess(
                make_env_fun = self.make_env_fun, 
                network_lock=mp.Lock(), 
                *self.args, **self.kwargs
            )
        }
        self._init_seed()

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
    
    def init_network(self):
        self.current_network = self.network_fun(self.dummy_env.observation_space.shape, self.dummy_env.action_space.n, *self.args, **self.kwargs).cuda().share_memory() 
        self.target_network  = self.network_fun(self.dummy_env.observation_space.shape, self.dummy_env.action_space.n, *self.args, **self.kwargs).cuda() 
        self.optimizer = self.optimizer_fun(self.current_network.parameters()) 
        self.update_target()
        
    def start_process(self):
        for _, process in self.process_dict.items():
            if isinstance(process, mp.Process):
                process.start()
            elif isinstance(process, list):
                for p in process:
                    p.start()

    def train(self):
        self.start_process()
        self.init_network()
        
        last_sim_steps_idx, ep_idx = 1, 1
        ep_reward_list = deque(maxlen=self.kwargs['ep_reward_avg_number'])
        tic   = time.time()
        self.process_dict['train_actor'].update_policy(network = self.current_network)
        for sim_steps_idx in range(1, self.kwargs['sim_steps'] + 1, self.kwargs['train_network_freq']):
            eps = self.line_schedule(sim_steps_idx-self.kwargs['train_start_step']) if sim_steps_idx > self.kwargs['train_start_step'] else 1
            data = self.process_dict['train_actor'].collect(steps_number = self.kwargs['train_network_freq'], eps=eps)
            for frames_idx, (action, obs, reward, done, info) in enumerate(data):
                self.process_dict['replay_buffer'].add(action, obs, reward, done)
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
                self.process_dict['eval_actor'].update_policy(network = deepcopy(self.current_network))
                self.process_dict['eval_actor'].eval(eval_idx = sim_steps_idx, eval_number = self.kwargs['eval_number'], eval_max_steps = self.kwargs['eval_max_steps'], eps = self.kwargs['eval_eps'])
                self.process_dict['eval_actor'].save_policy(name = sim_steps_idx)
                self.process_dict['eval_actor'].render(name=sim_steps_idx, render_max_steps=self.kwargs['eval_max_steps'], render_mode='rgb_array',fps=self.kwargs['eval_video_fps'], is_show=self.kwargs['eval_display'], eps = self.kwargs['eval_eps'])

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.process_dict['replay_buffer'].sample()
        with torch.no_grad():
            q_next = self.target_network(next_state).max(1)[0]
            q_target = reward + self.kwargs['gamma'] * (~done) * q_next 
        q = self.current_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = nn.MSELoss()(q_target, q)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.current_network.parameters(), self.kwargs['clip_gradient'])
        logger.add({'gradient_norm': gradient_norm.item(), 'loss': loss.item()})
        with self.network_lock:
            self.optimizer.step()
        return loss

class Nature_DQN_Async(Nature_DQN_Sync):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, actor_num=1, *args, **kwargs):
        super().__init__(make_env_fun, network_fun, optimizer_fun, *args, **kwargs)
        self.process_dict['train_actor'] = [
            NetworkActorProcess(
                make_env_fun = self.make_env_fun, 
                replay_buffer=self.process_dict['replay_buffer'], 
                network_lock=self.network_lock, 
                *self.args, **self.kwargs
            ) for _ in range(actor_num)
        ]

    def train(self):
        self.start_process()
        self.init_network()

        for actor in self.process_dict['train_actor']:
            actor.update_policy(network = self.current_network)

        while self.process_dict['replay_buffer'].check_size() < self.kwargs['train_start_step']:
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number = self.kwargs['train_network_freq'], sync=False, eps=1)

        for train_idx in range(1, self.kwargs['train_steps'] + 1):
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number = self.kwargs['train_network_freq'], sync=False, eps=self.line_schedule(train_idx), train_idx=train_idx)
            self.compute_td_loss()

            if (train_idx-1) % self.kwargs['train_update_target_freq'] == 0:
                self.update_target()

            if (train_idx-1) % self.kwargs['eval_freq'] == 0:
                self.process_dict['eval_actor'].update_policy(network = deepcopy(self.current_network))
                self.process_dict['eval_actor'].eval(eval_idx = train_idx, eval_number = self.kwargs['eval_number'], eval_max_steps = self.kwargs['eval_max_steps'], eps = self.kwargs['eval_eps'])
                self.process_dict['eval_actor'].save_policy(name = train_idx)
                self.process_dict['eval_actor'].render(name=train_idx, render_max_steps=self.kwargs['eval_max_steps'], render_mode='rgb_array',fps=self.kwargs['eval_video_fps'], is_show=self.kwargs['eval_display'], eps = self.kwargs['eval_eps'])

# %%
