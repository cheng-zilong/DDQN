import torch
import numpy as np
import random 
import time
from statistics import mean
from collections import deque
from utils.LogAsync import logger
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .Async import Async
from utils.Network import *
import os
import imageio
class ActorAsync(Async):
    STEP = 0
    EXIT = 1
    RESET = 2
    UPDATE_POLICY=3
    SAVE_POLICY=4
    EVAL=5
    RENDER=6
    UNWRAPPED_RESET=7
    def __init__(self, env, seed = None, *arg, **args):
        super().__init__(env, seed = None, *arg, **args)
        self.env = env
        self.seed = random.randint(0,10000) if seed == None else seed
        self.arg = arg 
        self.args = args 

    def run(self):
        self._init_seed()
        is_init_cache = False
        while True:
            (cmd, msg) = self._receive()
            if cmd == self.STEP:
                if not is_init_cache:
                    is_init_cache = True
                    self._send(self._collect(*(msg[0]), **(msg[1])))
                    self.cache = self._collect(*(msg[0]), **(msg[1]))
                else:
                    self._send(self.cache)
                    self.cache = self._collect(*(msg[0]), **(msg[1]))

            elif cmd == self.RESET:
                self._send(self._reset())
                is_init_cache = False

            elif cmd == self.UNWRAPPED_RESET:
                self._unwrapped_reset()
                is_init_cache = False

            elif cmd == self.EVAL:
                self._eval(*(msg[0]), **(msg[1]))

            elif cmd == self.UPDATE_POLICY:
                self._update_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.SAVE_POLICY:
                self._save_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.RENDER:
                self._render(*(msg[0]), **(msg[1]))

            elif cmd == self.EXIT:
                self._worker_pipe.close()
                return

            else:
                raise NotImplementedError

    def collect(self, steps_number, *arg, **args):
        args['steps_number'] = steps_number
        self.send(self.STEP, (arg, args))
        return self.receive()

    def reset(self):
        self.send(self.RESET, None)
        return self.receive()

    def unwrapped_reset(self):
        self.send(self.UNWRAPPED_RESET, None)
        return self.receive()

    def close(self):
        self.send(self.EXIT, None)
        self._pipe.close()

    def update_policy(self, *arg, **args):
        self.send(self.UPDATE_POLICY, (arg, args))

    def save_policy(self, *arg, **args):
        self.send(self.SAVE_POLICY, (arg, args))

    def eval(self, eval_idx = 0, *arg, **args):
        args['eval_idx'] = eval_idx
        self.send(self.EVAL, (arg, args))

    def render(self, *arg, **args):
        self.send(self.RENDER, (arg, args))

    def _init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.np_random.seed(self.seed)

    def _reset(self):
        self._actor_last_state = self.env.reset()
        self._actor_done_flag = False
        return self._actor_last_state

    def _unwrapped_reset(self):
        '''
        some gym envs cannot be reset because of the episodic life wrapper
        you can use this function to force the env to reset even it is not the end of an episode
        '''
        self.env.unwrapped.reset()
        self._actor_last_state = self.env.reset()
        self._actor_done_flag = False
        return self._actor_last_state

    def _collect(self, steps_number, *arg, **args):
        data = []
        for _ in range(steps_number):
            one_step = self._step(*arg, **args)
            data.append(one_step) 
        return data

    def _eval(self, eval_idx, *arg, **args):
        '''
        This is a template for the _eval method
        Overwrite this method to implement your own _eval method
        '''
        pass 

    def _step(self, *arg, **args):
        '''
        This is a template for the _step method
        Overwrite this method to implement your own _step method
        return [[action, state, reward, done, info], ...]
        '''
        pass 

    def _update_policy(self, *arg, **args):
        '''
        This is only a template for the _update_policy method
        Overwrite this method to implement your own _update_policy method
        '''
        pass

    def _save_policy(self, *arg, **args):
        '''
        This is only a template for the _save_policy method
        Overwrite this method to implement your own _save_policy method
        '''
        pass

    def _render(self, *arg, **args):
        '''
        This is only a template for the _render method
        Overwrite this method to implement your own _render method
        '''
        pass
class NetworkActorAsync(ActorAsync):
    '''
    Policy 是一个network的agent
    Policy 是从state到action的映射
    '''
    def __init__(self, env, network_lock, seed = None, *arg, **args):
        super().__init__(env, seed, *arg, **args)
        self._network_lock = network_lock
        self._network = None
        self._actor_done_flag = True
        self._actor_last_state = None

    def _step(self, eps, *arg, **args):
        '''
        epsilon greedy step
        '''
        if  self._network is None:
            raise Exception("Network has not been initialized!")
        # auto reset
        if self._actor_done_flag:
            self._actor_last_state = self.env.reset()
            self._actor_done_flag = False
            return [None, self._actor_last_state, None, None, None]
        eps_prob =  random.random()
        if eps_prob > eps:
            with self._network_lock:
                action = self._network.act(np.asarray(self._actor_last_state))
        else:
            action = self.env.action_space.sample()
        self._actor_last_state, reward, self._actor_done_flag, info = self.env.step(action)
        return action, self._actor_last_state, reward, self._actor_done_flag, info

    def _update_policy(self, network, *arg, **args):
        self._network = network

    def _save_policy(self, *arg, **args):
        '''
        name=
        idx is the name of this policy
        '''
        if not os.path.exists('save_model/' + logger._run_name + '/'):
            os.makedirs('save_model/' + logger._run_name + '/')
        torch.save(self._network.state_dict(), 'save_model/' + logger._run_name + '/' + str(args['name']) +'.pt')

    def _eval(self, eval_idx, eval_number, eval_max_steps, *arg, **args):
        ep_rewards_list = deque(maxlen=eval_number)
        for ep_idx in range(1, eval_number+1):
            self._unwrapped_reset()
            for ep_steps_idx in range(1, eval_max_steps + 1):
                _, _, _, done, info = self._collect(steps_number = 1, *arg, **args)[-1] 
                if done:
                    if info['episodic_return'] is not None: break
            ep_rewards_list.append(info['total_rewards'])
            ep_rewards_list_mean = mean(ep_rewards_list)
            logger.terminal_print(
                '--------(Evaluator Index: %d)'%(eval_idx), {
                '--------ep': ep_idx, 
                '--------ep_steps':  ep_steps_idx, 
                '--------ep_reward': info['total_rewards'], 
                '--------ep_reward_mean': ep_rewards_list_mean})
        logger.add({'eval_last': ep_rewards_list_mean})

    def _render(self, name, render_max_steps, render_mode, fps, is_show, figsize=(10, 5), dpi=160, *arg, **args):
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
            _, _, _, done, info = self._collect(steps_number = 1, *arg, **args)[-1] 
            ax.clear()
            ax.imshow(self.env.render(mode = render_mode))
            ax.axis('off')
            my_fig.canvas.draw()
            buf = my_fig.canvas.tostring_rgb()
            writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(fig_pixel_rows, fig_pixel_cols, 3))
            if done:
                if info['episodic_return'] is not None: break
        writer.close()


