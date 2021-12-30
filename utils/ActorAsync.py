import torch
import numpy as np
import random 
from gym import spaces
from statistics import mean
from collections import deque
from utils.LogAsync import logger
import matplotlib
import matplotlib.pyplot as plt
from .Async import Async
from utils.Network import *
import os
import imageio

class ActorAsync(Async):
    '''
    State must be np array
    done must be bool
    '''
    COLLECT = 0
    EXIT = 1
    RESET = 2
    UPDATE_POLICY=3
    SAVE_POLICY=4
    EVAL=5
    RENDER=6
    UNWRAPPED_RESET=7
    def __init__(self, env, seed = None, *args, **kwargs):
        super().__init__(env, seed = None, *args, **kwargs)
        self.env = env
        self.seed = random.randint(0,10000) if seed == None else seed
        self.__actor_state = None # It must be in a tenor format
        self.__actor_done_flag = True # Assume the initial state is done

    def run(self):
        self._init_seed()
        is_init_cache = False
        while True:
            (cmd, msg) = self._receive()
            if cmd == self.COLLECT:
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

    def collect(self, steps_number, *args, **kwargs):
        kwargs['steps_number'] = steps_number
        self.send(self.COLLECT, (args, kwargs))
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
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.np_random.seed(self.seed)

    def _reset(self):
        self.actor_state = self.env.reset()
        return self.actor_state

    def _unwrapped_reset(self):
        '''
        some gym envs cannot be reset because of the episodic life wrapper
        you can use this function to force the env to reset even it is not the end of an episode
        '''
        self.env.unwrapped.reset()
        self.actor_state = self.env.reset()
        return self.actor_state

    def _collect(self, steps_number, *args, **kwargs):
        data = []
        for _ in range(steps_number):
            one_step = self._step(*args, **kwargs)
            data.append(one_step) 
        return data

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
    def actor_done_flag(self):
        return self.__actor_done_flag

    @actor_done_flag.setter
    def actor_done_flag(self, value):
        if isinstance(value, bool):
            self.__actor_done_flag = value
        else:
            raise Exception('Done flag must be bool')

class NetworkActorAsync(ActorAsync):
    '''
    Policy 是一个network的agent
    Policy 是从state到action的映射
    '''
    def __init__(self, env, network_lock, seed = None, *args, **kwargs):
        super().__init__(env, seed)
        self._network_lock = network_lock
        self._network = None

    def _step(self, eps):
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
                _, _, _, done, info = self._collect(steps_number = 1, *args, **kwargs)[-1] 
                if (done) and (info is not None) and (info['episodic_return'] is not None): 
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
            _, _, _, done, info = self._collect(steps_number = 1, *args, **kwargs)[-1] 
            ax.clear()
            ax.imshow(self.env.render(mode = render_mode))
            ax.axis('off')
            my_fig.canvas.draw()
            buf = my_fig.canvas.tostring_rgb()
            writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(fig_pixel_rows, fig_pixel_cols, 3))
            if (done) and (info is not None) and (info['episodic_return'] is not None): 
                break
        writer.close()

class MultiPlayerSequentialGameNetworkActorAsync(ActorAsync):
    '''
    Multiple players
    Sequential Game
    Using neural network policy
    '''
    class _list_with_size:
        def __init__(self, size, init_element=None):
            self._data = list([init_element]*size)
            self._size = size

        def append(self, value):
            self._data.pop(0)
            self._data.append(value)

        def __getitem__(self, val):
            return self._data[val]
            
        def __str__(self):
            return str(self._data)

        def __array__(self):
            return np.asarray(self._data)

        def __eq__(self, other):
            return self.__array__() == other

        def __iter__(self):
            self.n = 0
            return self

        def __next__(self):
            if self.n < self._size:
                self.n += 1
                return self._data[self.n-1]
            else:
                raise StopIteration

    def __init__(self, env, network_lock, seed = None, player_number=2, *args, **kwargs):
        super().__init__(env, seed)
        self._player_number = player_number
        self._network_lock = network_lock
        self._network_list = deque(maxlen=player_number)
        self._actor_state_list = self._list_with_size(init_element=None, size=self._player_number+1) #多一个state存储最后一个玩家执行action后的状态
        self._actor_action_list = self._list_with_size(init_element=None, size=self._player_number*2) 
        self._actor_reward_list = self._list_with_size(init_element=[0]*self._player_number, size=self._player_number*2)
        self._actor_info_list = self._list_with_size(init_element=None, size=self._player_number+1) 
        self._not_done_number = 0
        self._done_state = None
        self._actor_done_list = self._list_with_size(init_element=None, size=self._player_number+1) 

    def _step(self, eps):
        '''
        epsilon greedy step
        return action_obs_reward_list, done, info
        每个player的一个完整step指的是，当前state采取action后到下一次需要采取action之前的state，产生的reward，done情况
        假设有5个player
        对于2号player，他返回的的完整step是：
            action: 2号player根据前一个state采取的action   (之后3、4、5、1号player根据自己的policy采取action)
            state: 2号player当前的state
            reward: 2号player两个state之间产生的对于2号player的reward之和
            done: 2号player两个state之间产生的done
            info: 2号player两个state之间产生的info
        
        reward是0-4 players采取action后reward的总和
        比如3号player采取的action对0号player有利，那么player0号的在当前次action后的reward会增加
        '''

        if len(self._network_list)==0:
            raise Exception("Network has not been initialized!")
        # auto reset
        if self.actor_done_flag:
            if self._not_done_number != 0:
                # 清空所有没reset的player，保证每一个player都有terminal state
                for _ in range(self._player_number):
                    self._actor_action_list.append(None) 
                    self._actor_state_list.append(self._done_state)
                    self._actor_reward_list.append([0]*self._player_number)
                    self._actor_done_list.append(True)
                    self._actor_info_list.append(None)
                self._not_done_number = 0
            else:
                self._actor_state_list.append(self._reset())
                for player_idx in range(self._player_number):
                    # 根据上一个state确定action
                    action = self._network_list[player_idx].eps_greedy_act(self._actor_state_list[-1], eps, self._network_lock, self.env.legal_action_mask)
                    self._actor_action_list.append(action) 
                    state, reward, self.actor_done_flag, info = self.env.step(action)
                    self._actor_state_list.append(np.asarray(state))
                    self._actor_reward_list.append(reward)
                    self._actor_done_list.append(False)
                    self._actor_info_list.append(info)
                    if self.actor_done_flag:
                        raise Exception("The game cannot be done during the reset period. Each player must take at least one step.")
        else:
            for player_idx in range(self._player_number):
                # 根据上一个state确定action
                action = self._network_list[player_idx].eps_greedy_act(self._actor_state_list[-1], eps, self._network_lock, self.env.legal_action_mask)
                self._actor_action_list.append(action)
                state, reward, self.actor_done_flag, info = self.env.step(action)
                self._actor_state_list.append(np.asarray(state))
                self._actor_reward_list.append(reward)
                self._actor_info_list.append(info)
                if self.actor_done_flag:
                    self._actor_done_list.append(True)
                    break
                else:
                    self._actor_done_list.append(False)
            if self.actor_done_flag:
                self._done_state = np.asarray(state)
                for _ in range(player_idx+1, self._player_number):
                    self._actor_action_list.append(None)
                    self._actor_state_list.append(self._done_state)
                    self._actor_reward_list.append([0]*self._player_number)
                    self._actor_done_list.append(True)
                    self._actor_info_list.append(None) 
                self._not_done_number = player_idx + 1
            # TODO some efficiency prblem with slicing with deque
        reward_array = np.asarray(self._actor_reward_list)
        return list(zip(self._actor_action_list[0:self._player_number], 
                        self._actor_state_list[0:self._player_number], 
                        [np.sum(reward_array[idx:idx+self._player_number,idx]) for idx in range(self._player_number)],
                        self._actor_done_list[0:self._player_number],
                        self._actor_info_list[0:self._player_number]))
            
    def _update_policy(self, network_list):
        for network in network_list:
            self._network_list.append(network)

    def _save_policy(self, name):
        '''
        name: The name of this policy
        '''
        if not os.path.exists('save_model/' + logger._run_name + '/'):
            os.makedirs('save_model/' + logger._run_name + '/')
        for player_idx, network in enumerate(self._network_list):
            torch.save(network.state_dict(), 'save_model/' + logger._run_name + '/' + str(name) +'_player%d.pt'%(player_idx))

    def _eval(self, eval_idx, eval_number, eval_max_steps, *arg, **args):
        ep_rewards_list = deque(maxlen=eval_number)
        for ep_idx in range(1, eval_number+1):
            self._unwrapped_reset()
            for ep_steps_idx in range(1, eval_max_steps + 1):
                # TODO 分开评估
                _, _, _, done, info = self._collect(steps_number = 1, *arg, **args)[-1][-1]
                if (done) and (info is not None) and (info['episodic_return'] is not None): break
            ep_rewards_list.append(info['total_rewards'])
            ep_rewards_list_mean = np.mean(np.array(ep_rewards_list),axis=0)
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
            # TODO 分开评估
            _, _, _, done, info = self._collect(steps_number = 1, *args, **kwargs)[-1][-1]
            ax.clear()
            ax.imshow(self.env.render(mode = render_mode), vmin=0, vmax=1)
            ax.axis('off')
            my_fig.canvas.draw()
            buf = my_fig.canvas.tostring_rgb()
            writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(fig_pixel_rows, fig_pixel_cols, 3))
            if (done) and (info is not None) and (info['episodic_return'] is not None): 
                break
        writer.close()