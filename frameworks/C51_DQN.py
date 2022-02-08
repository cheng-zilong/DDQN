import torch
import torch.nn as nn
from utils.Network import *
from utils.LogProcess import logger
from frameworks.Vanilla_DQN import Vanilla_DQN_Async
import torch.multiprocessing as mp
from utils.ActorProcess import C51_NetworkActorProcess
class C51_DQN_Async(Vanilla_DQN_Async):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs):
        super().__init__(make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs)
        self.process_dict['train_actor'] = [
            C51_NetworkActorProcess(
                make_env_fun = self.make_env_fun, 
                replay_buffer=self.process_dict['replay_buffer'], 
                network_lock=self.network_lock, 
                *self.args, **self.kwargs
            ) for _ in range(actor_num)
        ]
        self.process_dict['eval_actor'] = C51_NetworkActorProcess(
            make_env_fun = self.make_env_fun, 
            network_lock=mp.Lock(), 
            *self.args, **self.kwargs
        )

    def compute_td_loss(self):
        if not hasattr(self, 'delta_z'):
            self.delta_z = float(self.kwargs['v_max'] - self.kwargs['v_min']) / (self.kwargs['num_atoms'] - 1)
            self.torch_range = torch.arange(self.kwargs['batch_size'], dtype=torch.long, device='cuda:0')
        state, action, reward, next_state, done = self.process_dict['replay_buffer'].sample()
        with torch.no_grad():
            prob_next = self.target_network(next_state)
            q_next = (prob_next * self.current_network.atoms_gpu).sum(-1)
            a_next = torch.argmax(q_next, dim=-1)
            prob_next = prob_next[self.torch_range, a_next, :]
        rewards = reward.unsqueeze(-1)
        atoms_target = rewards + self.kwargs['gamma'] * (~done).unsqueeze(-1) * self.current_network.atoms_gpu.view(1, -1)
        atoms_target.clamp_(self.kwargs['v_min'], self.kwargs['v_max']).unsqueeze_(1)
        
        target_prob = (1 - (atoms_target - self.current_network.atoms_gpu.view(1, -1, 1)).abs() / self.delta_z).clamp(0, 1) * prob_next.unsqueeze(1)
        target_prob = target_prob.sum(-1)

        log_prob = self.current_network.forward_log(state)
        log_prob = log_prob[self.torch_range, action, :]
        loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.current_network.parameters(), self.kwargs['clip_gradient'])
        logger.add({'gradient_norm': gradient_norm.item(), 'loss': loss.item()})
        with self.network_lock:
            self.optimizer.step()
        return loss