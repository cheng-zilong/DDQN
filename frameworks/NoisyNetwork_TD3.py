import torch
import torch.nn as nn
from utils.Network import *
from utils.LogProcess import logger
from frameworks.Vanilla_TD3 import Vanilla_TD3_Async
from utils.ActorProcess import NoisyDDPG_NetworkActorProcess
import torch.multiprocessing as mp

class NoisyNetwork_TD3_Async(Vanilla_TD3_Async):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs):
        super().__init__(make_env_fun, network_fun, optimizer_fun, actor_num, *args, **kwargs)
        self.process_dict['train_actor'] = [
            NoisyDDPG_NetworkActorProcess(
                make_env_fun=self.make_env_fun, 
                replay_buffer=self.process_dict['replay_buffer'], 
                network_lock=self.network_lock, 
                *self.args, **self.kwargs
            ) for _ in range(actor_num)
        ]
        self.process_dict['eval_actor'] = NoisyDDPG_NetworkActorProcess(
            make_env_fun = self.make_env_fun, 
            network_lock=mp.Lock(), 
            *self.args, **self.kwargs
        )

    def init_network(self):
        self.current_network = self.network_fun(
            input_shape=self.dummy_env.observation_space.shape, 
            num_actions=self.dummy_env.action_space.shape[0], 
            num_critic=self.kwargs['network_critic_num'], 
            noise_std=self.kwargs['network_noise_std']
        ).cuda().share_memory() 
        self.target_network  = self.network_fun(
            input_shape=self.dummy_env.observation_space.shape, 
            num_actions=self.dummy_env.action_space.shape[0], 
            num_critic=self.kwargs['network_critic_num'], 
            noise_std=self.kwargs['network_noise_std']
        ).cuda() 
        self.optimizer_actor = self.optimizer_fun[0](self.current_network.policy_fc.parameters()) 
        self.optimizer_critic = self.optimizer_fun[1](self.current_network.value_fc.parameters()) 
        self.current_network.set_with_noise(True)
        self.target_network.set_with_noise(True)
        self.update_target(tau=1)

    def train(self):
        self.start_process()
        self.init_network()
        self.kwargs['actor_noise'] = 0
        self.kwargs['noise_clip'] = 0
        # 把action映射到-1到1之间
        self._action_multiplier = (2/(torch.tensor(self.dummy_env.action_space.high) - torch.tensor(self.dummy_env.action_space.low))).cuda()
        assert all(torch.isreal(self._action_multiplier)), 'some action_space range is illegal, the action_space must be finite'
        for actor in self.process_dict['train_actor']:
            actor.update_policy(network = self.current_network, with_noise=True)

        while self.process_dict['replay_buffer'].check_size() < self.kwargs['train_start_step']:
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number=self.kwargs['train_network_freq'], sigma=None)

        for train_idx in range(1, self.kwargs['train_steps'] + 1):
            for actor in self.process_dict['train_actor']:
                actor.collect(steps_number=self.kwargs['train_network_freq'], sigma=0)
            
            self.compute_td_loss()
            self.update_target(tau=self.kwargs['train_update_tau'])
            self.current_network.reset_noise()
            self.target_network.reset_noise()

            if (train_idx-1) % self.kwargs['eval_freq'] == 0:
                self.process_dict['eval_actor'].update_policy(network = self.current_network, with_noise=False)
                self.process_dict['eval_actor'].eval(
                    eval_idx=train_idx, 
                    eval_number=self.kwargs['eval_number'], 
                    eval_max_steps=self.kwargs['eval_max_steps'], 
                    sigma=0
                )
                self.process_dict['eval_actor'].save_policy(name = train_idx)
                self.process_dict['eval_actor'].render(
                    name=train_idx, 
                    render_max_steps=self.kwargs['eval_max_steps'], 
                    render_mode='rgb_array',
                    fps=self.kwargs['eval_video_fps'], 
                    is_show=self.kwargs['eval_display'], 
                    sigma = 0
                )

        