import torch
import numpy as np
import random 
from gym import spaces
from statistics import mean
from collections import deque
from utils.LogProcess import logger
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .BaseProcess import BaseProcess
from utils.Network import *
import os
import imageio
import time
from torch.multiprocessing import Value
from gym_envs.AtariWrapper import TotalRewardWrapper
class BaseActorProcess(BaseProcess):
    '''
    State must be np array
    done must be bool
    '''
    SYNC_COLLECT = 0
    EXIT = 1
    RESET = 2
    UPDATE_POLICY=3
    SAVE_POLICY=4
    EVAL=5
    RENDER=6
    UNWRAPPED_RESET=7
    ASYNC_COLLECT=8
    total_sim_steps = Value('i', 0)
    total_ep_num = Value('i', 0)
    actor_id = Value('i', 0)
    def __init__(self, make_env_fun, replay_buffer=None, *args, **kwargs):
        super().__init__(make_env_fun, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.env = make_env_fun(*args, **kwargs)
        self.replay_buffer = replay_buffer
        self.env.reset()
        self.seed = kwargs['seed']
        self.__actor_state = None # It must be in a tenor format
        self.__actor_reward = None
        self.__actor_done_flag = True # Assume the initial state is done

    def run(self):
        with BaseActorProcess.actor_id.get_lock():
            self.__actor_id = BaseActorProcess.actor_id.value
            BaseActorProcess.actor_id.value+=1
        self._init_seed()
        while True:
            (cmd, msg) = self._receive()
            if cmd == self.SYNC_COLLECT:
                self._sync_collect(*(msg[0]), **(msg[1]))

            elif cmd == self.ASYNC_COLLECT:
                self._async_collect(*(msg[0]), **(msg[1]))

            elif cmd == self.RESET:
                self._send(self._reset())

            elif cmd == self.UNWRAPPED_RESET:
                self._unwrapped_reset()

            elif cmd == self.EVAL:
                self._eval(*(msg[0]), **(msg[1]))

            elif cmd == self.UPDATE_POLICY:
                self._update_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.SAVE_POLICY:
                self._save_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.RENDER:
                self._render(*(msg[0]), **(msg[1]))

            elif cmd == self.EXIT:
                return

            else:
                raise NotImplementedError

    def is_env_done(self, done:bool, info):
        if hasattr(self.env, 'total_rewards'):
            return (info is not None) and (info['episodic_return'] is not None)
        else:
            return done

    def collect(self, steps_number, sync = True, *args, **kwargs):
        kwargs['steps_number'] = steps_number
        if sync:
            self.send(self.SYNC_COLLECT, (args, kwargs))
            return self.receive()
        else:
            self.send(self.ASYNC_COLLECT, (args, kwargs))
            return None

    def reset(self):
        self.send(self.RESET, None)
        return self.receive()

    def unwrapped_reset(self):
        self.send(self.UNWRAPPED_RESET, None)
        return self.receive()

    def close(self):
        self.send(self.EXIT, None)

    def update_policy(self, *args, **kwargs):
        self.send(self.UPDATE_POLICY, (args, kwargs))

    def save_policy(self, *args, **kwargs):
        self.send(self.SAVE_POLICY, (args, kwargs))

    def eval(self, eval_idx = 0, *args, **kwargs):
        kwargs['eval_idx'] = eval_idx
        self.send(self.EVAL, (args, kwargs))

    def render(self, *args, **kwargs):
        self.send(self.RENDER, (args, kwargs))

    def _init_seed(self):
        torch.manual_seed(self.seed+self.__actor_id)
        torch.cuda.manual_seed(self.seed+self.__actor_id)
        random.seed(self.seed+self.__actor_id)
        np.random.seed(self.seed+self.__actor_id)
        self.env.seed(self.seed+self.__actor_id)
        self.env.action_space.np_random.seed(self.seed+self.__actor_id)

    def _reset(self):
        self.actor_state = self.env.reset()
        if hasattr(self, '_cache'):
            del self._cache
        return self.actor_state

    def _unwrapped_reset(self):
        '''
        some gym envs cannot be reset because of the episodic life wrapper
        you can use this function to force the env to reset even it is not the end of an episode
        '''
        self.env.unwrapped.reset()
        self.actor_state = self.env.reset()
        if hasattr(self, '_cache'):
            del self._cache
        return self.actor_state

    def _sync_collect_helper(self, steps_number, *args, **kwargs):
        data_list = []
        for _ in range(steps_number):
            one_step_data = self._step(*args, **kwargs)
            data_list.append(one_step_data) 
        return data_list

    def _sync_collect(self, steps_number, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._send(self._sync_collect_helper(steps_number, *args, **kwargs))
            self._cache = self._sync_collect_helper(steps_number, *args, **kwargs)
        else:
            self._send(self._cache)
            self._cache = self._sync_collect_helper(steps_number, *args, **kwargs)
        
    def _async_collect(self, steps_number, *args, **kwargs):
        if not hasattr(self, '_ep_reward_list'):
            self._ep_reward_list = deque(maxlen=self.kwargs['ep_reward_avg_number'])
            self._sim_steps_idx = 0
            self._last_sim_steps_idx = 0
            self._ep_tic = time.time()
        for frames_idx in range(steps_number):
            action, obs, reward, done, info = self._step(*args, **kwargs)
            self.replay_buffer.add(action, obs, reward, done, self.__actor_id)
            self._sim_steps_idx += 1
            BaseActorProcess.total_sim_steps.value += 1
            if self.is_env_done(done, info): 
                _episodic_steps = self._sim_steps_idx + frames_idx - self._last_sim_steps_idx
                self._ep_reward_list.append(info['episodic_return'])
                BaseActorProcess.total_ep_num.value += 1
                logger_dict = {
                    'ep': BaseActorProcess.total_ep_num.value, 
                    'ep_steps': _episodic_steps, 
                    'ep_reward': info['episodic_return'], 
                    'ep_reward_avg': mean(self._ep_reward_list), 
                    'fps': _episodic_steps / (time.time()-self._ep_tic), 
                    'queue_buffer_size': self._queue.qsize()
                }
                self._ep_tic = time.time()
                logger.add({**logger_dict, **kwargs})
                # 每个agent 完成一个scenario后print，不通过算法控制print
                logger.wandb_print('(Training Agent %d)'%(self.__actor_id), step=BaseActorProcess.total_sim_steps.value)
                self._last_sim_steps_idx = self._sim_steps_idx + frames_idx

    def _eval(self, eval_idx, *args, **kwargs):
        '''
        This is a template for the _eval method
        Overwrite this method to implement your own _eval method
        '''
        pass 

    def _step(self, *args, **kwargs):
        '''
        This is a template for the _step method
        Overwrite this method to implement your own _step method
        return [[action, state, reward, done, info], ...]
        '''
        pass 

    def _update_policy(self, *args, **kwargs):
        '''
        This is only a template for the _update_policy method
        Overwrite this method to implement your own _update_policy method
        '''
        pass

    def _save_policy(self, *args, **kwargs):
        '''
        This is only a template for the _save_policy method
        Overwrite this method to implement your own _save_policy method
        '''
        pass

    def _render(self, *args, **kwargs):
        '''
        This is only a template for the _render method
        Overwrite this method to implement your own _render method
        '''
        pass

    @property
    def actor_state(self):
        return self.__actor_state

    @actor_state.setter
    def actor_state(self, value):
        self.__actor_state = np.asarray(value)

    @property
    def actor_reward(self):
        return self.__actor_reward

    @actor_reward.setter
    def actor_reward(self, value):
        self.__actor_reward = np.asarray(value)

    @property
    def actor_done_flag(self):
        return self.__actor_done_flag

    @actor_done_flag.setter
    def actor_done_flag(self, value):
        if isinstance(value, bool):
            self.__actor_done_flag = value
        else:
            raise Exception('Done flag must be bool')

class NetworkActorProcess(BaseActorProcess):
    '''
    Policy 是一个network的agent
    Policy 是从state到action的映射
    '''
    def __init__(self, make_env_fun, network_lock, replay_buffer = None, *args, **kwargs):
        super().__init__(make_env_fun, replay_buffer, *args, **kwargs)
        self._network_lock = network_lock
        self._network = None

    def _step(self, eps, *args, **kwargs):
        '''
        epsilon greedy step
        '''
        if self._network is None:
            raise Exception("Network has not been initialized!")
        if self.actor_done_flag:
            # auto reset
            self.actor_state = self._reset()
            self.actor_done_flag = False
            return None, self.actor_state, None, None, None
        action = self._network.eps_greedy_act(self.actor_state, eps, self._network_lock)

        self.actor_state, reward, self.actor_done_flag, info = self.env.step(action)
        return action, self.actor_state, reward, self.actor_done_flag, info

    def _update_policy(self, network):
        self._network = network

    def _save_policy(self, name):
        '''
        The name of this policy
        '''
        if not os.path.exists('save_model/' + logger._run_name + '/'):
            os.makedirs('save_model/' + logger._run_name + '/')
        torch.save(self._network.state_dict(), 'save_model/' + logger._run_name + '/' + str(name) +'.pt')

    def _eval(self, eval_idx, eval_number, eval_max_steps, *args, **kwargs):
        ep_rewards_list = deque(maxlen=eval_number)
        for ep_idx in range(1, eval_number+1):
            self._unwrapped_reset()
            for ep_steps_idx in range(1, eval_max_steps + 1):
                _, _, _, done, info = self._sync_collect_helper(steps_number = 1, *args, **kwargs)[-1] 
                if self.is_env_done(done, info):
                    break
            ep_rewards_list.append(info['total_rewards'])
            ep_rewards_list_mean = mean(ep_rewards_list)
            logger.terminal_print(
                '--------(Evaluator Index: %d)'%(eval_idx), {
                '--------ep': ep_idx, 
                '--------ep_steps':  ep_steps_idx, 
                '--------ep_reward': info['total_rewards'], 
                '--------ep_reward_mean': ep_rewards_list_mean})
        logger.add({'eval_last': ep_rewards_list_mean})

    def _render(self, name, render_max_steps, render_mode, fps, is_show, figsize=(10, 5), dpi=160, *args, **kwargs):
        if not is_show: matplotlib.use('Agg')
        if not os.path.exists('save_video/' + logger._run_name + '/'):
            os.makedirs('save_video/' + logger._run_name + '/')
        writer = imageio.get_writer('save_video/' + logger._run_name + '/' + str(name) +'.mp4', fps = fps)
        my_fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.rcParams['font.size'] = '8'
        ax = my_fig.add_subplot(111)
        my_fig.tight_layout()
        fig_pixel_cols, fig_pixel_rows = my_fig.canvas.get_width_height()
        self._unwrapped_reset()
        for _ in range(1, render_max_steps + 1):
            _, _, _, done, info = self._sync_collect_helper(steps_number = 1, *args, **kwargs)[-1] 
            ax.clear()
            ax.imshow(self.env.render(mode = render_mode))
            ax.axis('off')
            my_fig.canvas.draw()
            buf = my_fig.canvas.tostring_rgb()
            writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(fig_pixel_rows, fig_pixel_cols, 3))
            if self.is_env_done(done, info):
                break
        writer.close()

class C51_NetworkActorProcess(NetworkActorProcess):
    def _render(self, name, render_max_steps, render_mode, fps, is_show, figsize=(10, 5), dpi=160, *args, **kwargs):
        if not is_show: matplotlib.use('Agg')
        if not os.path.exists('save_video/' + logger._run_name + '/'):
            os.makedirs('save_video/' + logger._run_name + '/')
        writer = imageio.get_writer('save_video/' + logger._run_name + '/' + str(name) +'.mp4', fps = fps)
        my_fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(1, 2)
        ax_left, ax_right = my_fig.add_subplot(gs[0]), my_fig.add_subplot(gs[1])
        my_fig.tight_layout()
        fig_pixel_cols, fig_pixel_rows = my_fig.canvas.get_width_height()
        self._unwrapped_reset()
        for _ in range(1, render_max_steps + 1):
            action, _, _, done, info = self._sync_collect_helper(steps_number = 1, *args, **kwargs)[-1] 
            action_prob = np.swapaxes(self._network.action_prob[0].cpu().numpy(),0, 1)
            legends = []
            for i, action_meaning in enumerate(self.env.unwrapped.get_action_meanings()):
                legend_text = ' (Q=%+.2e)'%(self._network.action_Q[0,i]) if i == action else ' (Q=%+.2e)*'%(self._network.action_Q[0,i])
                legends.append(action_meaning + legend_text) 
            ax_left.clear()
            ax_left.imshow(self.env.render(mode = render_mode))
            ax_left.axis('off')
            ax_right.clear()
            ax_right.plot(self._network.atoms_cpu, action_prob)
            ax_right.legend(legends)
            ax_right.grid(True)
            my_fig.canvas.draw()
            buf = my_fig.canvas.tostring_rgb()
            writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(fig_pixel_rows, fig_pixel_cols, 3))
            if self.is_env_done(done, info): 
                break
        writer.close()