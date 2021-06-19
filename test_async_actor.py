
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
from wrapper import make_atari, wrap_deepmind, wrap_pytorch, OriginalReturnWrapper
from statistics import mean
from ReplayBufferAsync import ReplayBufferAsync
import copy 
import torch.multiprocessing as mp
from ActorAsync import ActorAsync

class DQN:
    def __init__(self, env, netowrk, optimizer, *arg, **args):
        self.env = env
        self.batch_size=args['batch_size']
        self.gamma=args['gamma']
        self.gradient_clip = args['gradient_clip']
        self.eps_start = args['eps_start']
        self.eps_end = args['eps_end']
        self.eps_decay_steps = args['eps_decay_steps']
        self.total_steps = args['total_steps']
        self.start_training_steps = args['start_training_steps']
        self.update_target_freq = args['update_target_freq']
        self.train_freq = args['train_freq']
        self.seed = args['seed']

        self.lock = mp.Lock()
        self.actor = ActorAsync(self.env, seed=self.seed, lock=self.lock)
        self.replay_buffer = ReplayBufferAsync(args['buffer_size'], args['batch_size'], cache_size=2, seed = self.seed)
        self.current_model = netowrk(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        self.current_model.share_memory()
        self.target_model  = netowrk(self.env.observation_space.shape, self.env.action_space.n, **args).cuda()
        self.optimizer = optimizer(self.current_model.parameters())
        self.actor.set_network(self.current_model)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def train(self):
        last_steps_idx, ep_idx = 1, 1
        ep_reward_10_list = deque(maxlen=10)
        state = self.env.reset()
        loss  = torch.tensor(0) 
        tic   = time.time()
        fps   = 0
        for steps_idx in range(1, self.total_steps + 1):
            eps = self.obtain_eps(steps_idx-self.start_training_steps) if steps_idx > self.start_training_steps else 1
            state, action, reward, next_state, done, info = self.actor.step(eps)
            self.replay_buffer.add(state, action, reward, next_state, done)

            if info['episodic_return'] is not None:
                episodic_steps = steps_idx-last_steps_idx
                ep_reward_10_list.append(info['episodic_return'])
                toc = time.time()
                fps = episodic_steps/(toc-tic)
                tic = time.time()
                ep_reward_10_list_mean = 0 if len(ep_reward_10_list) == 0 else mean(ep_reward_10_list)

                wandb.log({'ep_reward': info['episodic_return'], 'ep_reward_avg': ep_reward_10_list_mean, 'loss': loss, 'eps': eps, 'fps': fps}, step=steps_idx)
                print('(Training Agent)', end =" ") if steps_idx > self.start_training_steps else print('(Collecting Data)', end =" ")
                print('ep=%6d ep_reward_last=%.2f ep_reward_avg=%.2f ep_steps=%4d total_steps=%7d loss=%.4f eps=%.4f fps=%.2f '%
                        (ep_idx, info['episodic_return'], ep_reward_10_list_mean, episodic_steps, steps_idx, loss.item(), eps, fps))
                ep_idx += 1
                last_steps_idx = steps_idx
            if steps_idx > self.start_training_steps and steps_idx % self.train_freq == 0:
                loss = self.compute_td_loss()
                pass
            if steps_idx / self.train_freq % self.update_target_freq == 0:
                self.update_target() 
                
    def test(self):
        pass

    def learn(self):
        self.train()
        torch.save(self.current_model.state_dict(), self.__class__.__name__ + '(' + self.env.unwrapped.spec.id + ')_' + str(self.seed) + '.pt')
        
    def obtain_eps(self, steps_idx):
        eps = self.eps_end + (self.eps_start - self.eps_end) * (1 - min(steps_idx,self.eps_decay_steps) / self.eps_decay_steps)
        return eps

class CatDQN(DQN):
    def __init__(self, env, network, optimizer, *arg, **args):
        super().__init__(env, network, optimizer, *arg, **args)
        self.num_atoms = args['num_atoms']
        self.v_min = args['v_min']
        self.v_max = args['v_max']
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).cuda()
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).cuda()
        self.torch_range = torch.arange(self.batch_size).long().cuda()

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

        log_prob = self.current_model(state).log()
        # log_prob = log_prob[self.torch_range, action, :]
        # loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1).mean()

        # #### for debug pruposes
        # if torch.isnan(loss):
        #     print('.....')
        # self._last_state = state
        # self._last_action = action
        # self._last_reward = reward
        # self._last_next_state = next_state
        # self._last_done = done
        # self._last_model_para =  copy.deepcopy(self.current_model.state_dict())

        # self.optimizer.zero_grad()
        # loss.backward()
        # nn.utils.clip_grad_norm_(self.current_model.parameters(), self.gradient_clip)
        # with self.lock:
        #     self.optimizer.step()

        # return loss
        return torch.tensor(0)

if __name__ == '__main__':
    parser = get_default_parser()
    parser.set_defaults(seed=4) 
    parser.set_defaults(env_name= 'BreakoutNoFrameskip-v4')
    parser.set_defaults(total_steps = int(1e7))
    parser.set_defaults(start_training_steps=1000)
    # parser.set_defaults(start_training_steps=1000)
    parser.set_defaults(train_freq=4)
    parser.set_defaults(update_target_freq=10000)
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.)
    parser.add_argument('--v_max', type=float, default=10.)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env    = make_atari(args.env_name)
    env    = OriginalReturnWrapper(env)
    env    = wrap_deepmind(env, frame_stack=True)
    env    = wrap_pytorch(env)
    env.seed(args.seed)
    env.action_space.np_random.seed(args.seed)
    wandb.init(name='CatCnnDQN(' + args.env_name + ')_' + str(args.seed), project="C51", config=args)
    CatDQN(
        env=env, 
        network = CatCnnQNetwork, 
        optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, eps=args.opt_eps),  
        **vars(args)
        ).learn()

    

# %%
