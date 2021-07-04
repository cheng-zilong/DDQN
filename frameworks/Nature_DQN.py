
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
from utils.EvaluationAsync import EvaluationAsync

class Nature_DQN:
    def __init__(self, make_env_fun, netowrk_fun, optimizer_fun, *arg, **args):
        self.arg = arg 
        self.args = args 
        self.env = make_env_fun(**args)
        self.gamma=args['gamma']
        self.gradient_clip = args['gradient_clip']
        self.eps_start = args['eps_start']
        self.eps_end = args['eps_end']
        self.eps_decay_steps = args['eps_decay_steps']
        self.start_training_steps = args['start_training_steps']
        self.update_target_steps = args['update_target_steps']
        self.eval_freq = args['eval_freq']

        self.network_lock = mp.Lock()
        self.actor = ActorAsync(env = self.env, network_lock = self.network_lock, *arg, **args)
        self.replay_buffer = ReplayBufferAsync(*arg, **args)
        self.evaluator = EvaluationAsync(make_env_fun = make_env_fun, **args)

        self.current_network = netowrk_fun(self.env.observation_space.shape, self.env.action_space.n, **args).cuda().share_memory()
        self.target_network  = netowrk_fun(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        self.optimizer = optimizer_fun(self.current_network.parameters())
        self.update_target()
        
        self.evaluator.init(netowrk_fun)
        
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
        self.actor.set_network(self.current_network)
        for train_steps_idx in range(1, self.args['train_steps'] + 1, self.args['train_freq']):
            eps = self.line_schedule(train_steps_idx-self.start_training_steps) if train_steps_idx > self.start_training_steps else 1
            data = self.actor.step(eps)
            for frames_idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs[None,-1], reward, done)
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

            if (train_steps_idx-1) % self.update_target_steps == 0:
                self.update_target()
                
            if (train_steps_idx-1) % self.eval_freq == 0:
                self.evaluator.eval(train_steps=train_steps_idx, state_dict=self.current_network.state_dict())


# %%
