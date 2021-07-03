
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

class DQN:
    def __init__(self, make_env_fun, netowrk_fun, optimizer_fun, *arg, **args):
        self.arg = arg 
        self.args = args 
        self.env = make_env_fun(args['env_name'], **args)
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
        self.evaluator = EvaluationAsync(env = self.env, **args)

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

class CatDQN(DQN):
    def __init__(self, make_env, network, optimizer, *arg, **args):
        super().__init__(make_env, network, optimizer, *arg, **args)
        self.v_min = args['v_min']
        self.v_max = args['v_max']
        self.delta_z = float(self.v_max - self.v_min) / (args['num_atoms'] - 1)
        self.atoms_gpu = torch.linspace(self.v_min, self.v_max, args['num_atoms']).cuda()
        self.offset = torch.linspace(0, (args['batch_size'] - 1) * args['num_atoms'], args['batch_size']).long().unsqueeze(1).expand(args['batch_size'], args['num_atoms']).cuda()
        self.torch_range = torch.arange(args['batch_size']).long().cuda()

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
        with self.network_lock:
            self.optimizer.step()

        return loss

    

# %%
