
#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
from logging import log
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
matplotlib.use('Agg')
class DQN:
    def __init__(self, make_env, netowrk, optimizer, *arg, **args):
        self.arg = arg 
        self.args = args 
        self.env = make_env(args['env_name'], **args)
        self.batch_size=args['batch_size']
        self.gamma=args['gamma']
        self.gradient_clip = args['gradient_clip']
        self.eps_start = args['eps_start']
        self.eps_end = args['eps_end']
        self.eps_decay_steps = args['eps_decay_steps']
        self.total_steps = args['total_steps']
        self.start_training_steps = args['start_training_steps']
        self.update_target_steps = args['update_target_steps']
        self.train_freq = args['train_freq']
        self.seed = args['seed']
        self.save_model_steps = args['save_model_steps']

        self.lock = mp.Lock()
        self.actor = ActorAsync(self.env, steps_no=self.train_freq, seed=self.seed, lock=self.lock)
        self.replay_buffer = ReplayBufferAsync(args['buffer_size'], args['batch_size'], seed = self.seed, stack_frames=self.env.observation_space.shape[0])
        self.current_model = netowrk(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        self.current_model.share_memory()
        self.target_model  = netowrk(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        self.optimizer = optimizer(self.current_model.parameters())
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def train(self):
        last_steps_idx, ep_idx = 1, 1
        ep_reward_10_list = deque(maxlen=10)
        loss  = torch.tensor(0) 
        tic   = time.time()
        now = datetime.now()
        self.actor.set_network(self.current_model)
        for steps_idx in range(1, self.total_steps + 1, self.train_freq):
            eps = self.obtain_eps(steps_idx-self.start_training_steps) if steps_idx > self.start_training_steps else 1
            data = self.actor.step(eps)
            for idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs[None,-1], reward, done)
                if info is not None and info['episodic_return'] is not None:
                    episodic_steps = steps_idx + idx - last_steps_idx
                    ep_reward_10_list.append(info['episodic_return'])
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    ep_reward_10_list_mean = mean(ep_reward_10_list)

                    logger.add({'total_steps':steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': info['episodic_return'], 'ep_reward_avg': ep_reward_10_list_mean, 'loss': loss.item(), 'eps': eps, 'fps': fps})
                    logger.print('(Training Agent) ', step=steps_idx) if steps_idx > self.start_training_steps else logger.print('(Collecting Data) ', step=steps_idx)
                    ep_idx += 1
                    last_steps_idx = steps_idx + idx

            if steps_idx > self.start_training_steps:
                loss = self.compute_td_loss()

            if steps_idx % self.update_target_steps == 1:
                self.update_target() 

            if steps_idx % self.save_model_steps == 1 and steps_idx != 1:
                torch.save(self.current_model.state_dict(), 'save_model/' + self.__class__.__name__ + '(' + self.env.unwrapped.spec.id + ')_' + str(self.seed) + '_' + now.strftime("%Y%m%d-%H%M%S") + '.pt')
                
            if steps_idx % 50000 == 1:
                self.logger_samples()

    def learn(self):
        pass

    def eval(self):
        #python test_async_actor.py --mode eval --model_path "save_model/CatDQN(BreakoutNoFrameskip-v4)_4_20210621-024725.pt" --seed 6  
        model_path = self.args['model_path']
        self.current_model.load_state_dict(torch.load(model_path))
        state = self.env.reset()
        ep_idx = 1
        ep_reward_10_list = deque(maxlen=10)
        tic   = time.time()
        last_steps_idx = 1
        for steps_idx in range(1, self.total_steps + 1):
            action = self.current_model.act(state)
            state, _, done, info = self.env.step(action)
            self.env.render()
            if done:
                state = self.env.reset()
                if info['episodic_return'] is not None:
                    episodic_steps = steps_idx - last_steps_idx
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    ep_idx+=1
                    ep_reward_10_list.append(info['episodic_return'])
                    ep_reward_10_list_mean = mean(ep_reward_10_list)
                    logger.add({'total_steps':steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': info['episodic_return'], 'ep_reward_avg': ep_reward_10_list_mean, 'fps': fps})
                    logger.print('(Training Agent)', step=steps_idx)
                    last_steps_idx = steps_idx
        
    def obtain_eps(self, steps_idx):
        eps = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(steps_idx,self.eps_decay_steps) / self.eps_decay_steps)
        return eps

    def logger_samples(self):
        image = self.render()
        logger.add({'sample_images': wandb.Image(image)})

    def render(self):
        pass 

class CatDQN(DQN):
    def __init__(self, make_env, network, optimizer, *arg, **args):
        super().__init__(make_env, network, optimizer, *arg, **args)
        self.num_atoms = args['num_atoms']
        self.v_min = args['v_min']
        self.v_max = args['v_max']
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).cuda()
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).cuda()
        self.torch_range = torch.arange(self.batch_size).long().cuda()
        self.my_fig, self.my_axes = plt.subplots(1, 2, figsize = (20, 10))
        self.my_fig.suptitle('Samples')
        
    def render(self):
        self.my_axes[1].cla()
        with torch.no_grad():
            state, _, _, _, _ = self.replay_buffer.sample()
            prob_next = self.current_model(state)
        self.my_axes[0].imshow(state[-1,-1].cpu().numpy())
        self.my_axes[1].plot(self.atoms.cpu().numpy(), np.swapaxes(prob_next[0].cpu().numpy(),0,1))
        self.my_axes[1].legend(self.env.unwrapped.get_action_meanings())
        self.my_axes[1].grid(True)
        return self.my_fig

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            prob_next = self.target_model(next_state)
            q_next = (prob_next * self.atoms).sum(-1)
            a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.torch_range, a_next, :]

        rewards = reward.unsqueeze(-1)
        atoms_target = rewards + self.gamma * (~done).unsqueeze(-1) * self.atoms.view(1, -1)
        atoms_target.clamp_(self.v_min, self.v_max).unsqueeze_(1)
        
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_z).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.current_model.forward_log(state)
        log_prob = log_prob[self.torch_range, action, :]
        loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.current_model.parameters(), self.gradient_clip)
        gradient_norm = nn.utils.clip_grad_norm_(self.current_model.parameters(), self.gradient_clip)
        logger.add({'gradient_norm': gradient_norm.item()})
        with self.lock:
            self.optimizer.step()

        return loss

if __name__ == '__main__':
    parser = get_default_parser()
    parser.set_defaults(seed=777) 
    parser.set_defaults(env_name= 'BreakoutNoFrameskip-v4')
    # parser.set_defaults(env_name= 'SpaceInvadersNoFrameskip-v4')
    # parser.set_defaults(env_name= 'PongNoFrameskip-v4')
    parser.set_defaults(total_steps = int(5e7))
    parser.set_defaults(start_training_steps=50000)
    # parser.set_defaults(start_training_steps=1000)
    parser.set_defaults(gradient_clip = 10)
    
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.)
    parser.add_argument('--v_max', type=float, default=10.)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        logger.init(project_name='C51', args=args)
        CatDQN(
            make_env = make_env,
            network = CatCnnQNetwork, 
            optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).train()
    elif args.mode == 'eval':
        logger.init()
        CatDQN(
            make_env = make_env,
            network = CatCnnQNetwork, 
            optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
            **vars(args)
            ).eval()

    

# %%
