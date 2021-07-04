
#%%
'''
layer_init + 4 step 1 gradient + async buffer
'''
import torch
import torch.nn as nn
from utils.Network import *
from utils.LogAsync import logger
from frameworks.Nature_DQN import Nature_DQN

class C51_DQN(Nature_DQN):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *arg, **args):
        super().__init__(make_env_fun, network_fun, optimizer_fun, *arg, **args)
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
