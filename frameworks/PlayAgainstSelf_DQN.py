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
from utils.EvaluatorAsync import EvaluatorAsync
from .Nature_DQN import Nature_DQN
import random

class PlayAgainstSelf_DQN(Nature_DQN):
    def __init__(self, make_env_fun, network_fun, optimizer_fun, *arg, **args):
        self.network_lock = mp.Lock()
        self.env = make_env_fun(**args)
        self.actor = TwoPlayer_ActorAsync(env = self.env, network_lock = self.network_lock, *arg, **args)
        self.evaluator = TwoPlayer_EvaluationAsync(make_env_fun = make_env_fun, **args)
        super().__init__(make_env_fun, network_fun, optimizer_fun, *arg, **args)
        
    def train(self):
        self.evaluator.init(self.network_fun)
        last_train_steps_idx, ep_idx = 1, 1
        ep_reward_list = deque(maxlen=self.args['ep_reward_avg_number'])
        loss  = torch.tensor(0)
        tic   = time.time()
        self.actor.set_network(self.current_network)
        for train_steps_idx in range(1, self.args['train_steps'] + 1, self.args['train_freq']):
            eps = self.line_schedule(train_steps_idx-self.start_training_steps) if train_steps_idx > self.start_training_steps else 1
            data = self.actor.collect(eps)
            for frames_idx, (action, obs, reward, done, info) in enumerate(data):
                self.replay_buffer.add(action, obs, reward, done)
                if info is not None and info['episodic_return'] is not None:
                    episodic_steps = train_steps_idx + frames_idx - last_train_steps_idx
                    if info['winner']==1:
                        ep_reward_list.append(info['episodic_return'])
                    elif info['winner']==0:
                        ep_reward_list.append(0)
                    else:
                        ep_reward_list.append(-info['episodic_return'])
                    toc = time.time()
                    fps = episodic_steps / (toc-tic)
                    tic = time.time()
                    logger.add({'train_steps':train_steps_idx ,'ep': ep_idx, 'ep_steps': episodic_steps, 'ep_reward': ep_reward_list[-1], 'ep_reward_avg': mean(ep_reward_list), 'loss': loss.item(), 'eps': eps, 'fps': fps})
                    logger.wandb_print('(Training Agent) ', step=train_steps_idx) if train_steps_idx > self.start_training_steps else logger.wandb_print('(Collecting Data) ', step=train_steps_idx)
                    ep_idx += 1
                    last_train_steps_idx = train_steps_idx + frames_idx

            if train_steps_idx > self.start_training_steps:
                loss = self.compute_td_loss()

            if (train_steps_idx-1) % self.update_target_steps == 0:
                self.update_target()
                
            if (train_steps_idx-1) % self.eval_freq == 0:
                self.evaluator.eval(eval_idx=train_steps_idx, state_dict=self.current_network.state_dict())

class TwoPlayer_ActorAsync(ActorAsync):
    def eps_greedy_step(self, eps):
        # auto reset
        data = []
        for _ in range(self.steps_no):
            if self.done:
                self.state = self.env.reset()
                data.append([None, self.state, None, None, None])
                self.done = False
                continue
            if random.random() > eps:
                with self.network_lock: action = self._network.act(np.asarray(self.state))
            else:
                action = self.env.action_space.sample()
            obs, reward, self.done, info = self.env.step(action)
            if info['illegal_move'] or self.done:
                data.append([action, obs, reward, self.done, info])
                continue
            else:
                while True:
                    if random.random() > eps:
                        with self.network_lock: opponent_action = self._network.act(-np.asarray(obs))
                    else:
                        opponent_action = self.env.action_space.sample()
                    obs, reward, self.done, info = self.env.step(opponent_action)
                    if not info['illegal_move']:
                        break
                if self.done: #说明对手赢了
                    reward = -reward
                data.append([action, obs, reward, self.done, info])
                self.state = obs
        return data

class TwoPlayer_EvaluationAsync(EvaluatorAsync):
    def _eval(self, ep_idx):
        env = self.make_env_fun(**self.args)
        env.seed(self.seed+ep_idx)
        env.action_space.np_random.seed(self.seed+ep_idx)
        state = env.reset()
        next_player = 1
        tic   = time.time()
        for ep_steps_idx in range(1, self.args['eval_max_steps'] + 1):
            eps_prob =  random.random()
            action = self.evaluator_policy.act(np.asarray(state) if next_player==1 else -np.asarray(state)) if eps_prob > self.eval_eps else env.action_space.sample()
            state, _, done, info = env.step(action)
            next_player = info['next_player']
            if (ep_idx is None or ep_idx in self.eval_render_save_video) and \
                (ep_steps_idx-1) % self.eval_render_freq == 0 : # every eval_render_freq frames sample 1 frame
                if self.args['eval_display']: 
                    env.render(folder = self.gif_folder, number = self.current_train_steps)
            if done:
                state = env.reset()
                if info['episodic_return'] is not None: break
        toc = time.time()
        fps = ep_steps_idx / (toc-tic)
        return ep_steps_idx, info['total_rewards'], fps

# %%

