from cv2 import data
import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import random

from baselines.deepq.replay_buffer import ReplayBuffer


class ReplayBufferTorch(ReplayBuffer):
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return torch.stack(obses_t), torch.stack(actions), torch.stack(rewards), torch.stack(obses_tp1), torch.stack(dones)

class ReplayBufferAsync(mp.Process):
    '''
    add numpy
    sample torch.tensor.cuda()
    '''
    ADD = 0
    SAMPLE = 1
    CLOSE = 2

    def __init__(self, buffer_size, batch_size, cache_size, seed):
        mp.Process.__init__(self)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.is_init_cache = False
        self.current_cache_size = self.cache_size # output pointer 0 when initilize the cache
        self.out_pointer = 1 # output pointer 0 when initialize, 1 when first output
        self.in_pointer = 0 # update pointer 0 when first update
        self.seed = seed
        self.start()

    def run(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        replay_buffer = ReplayBuffer(self.buffer_size)
        memory_share_list = []
        
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.ADD:
                replay_buffer.add(*data)
            elif op == self.SAMPLE:
                if not self.is_init_cache:
                    self.is_init_cache=True
                    state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                    state_share = torch.zeros((self.cache_size, *state.shape), dtype=torch.tensor(state).dtype, device=torch.device(0)).share_memory_()
                    action_share = torch.zeros((self.cache_size, *action.shape), dtype=torch.tensor(action).dtype, device=torch.device(0)).share_memory_()
                    reward_share = torch.zeros((self.cache_size, *reward.shape), dtype=torch.tensor(reward).dtype, device=torch.device(0)).share_memory_()
                    next_state_share = torch.zeros((self.cache_size, *next_state.shape), dtype=torch.tensor(next_state).dtype, device=torch.device(0)).share_memory_()
                    done_share = torch.zeros((self.cache_size, *done.shape), dtype=torch.tensor(done).dtype, device=torch.device(0)).share_memory_()
                    for i in range(0, self.cache_size): # no need update 0
                        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                        state_share[i] = torch.tensor(state, device=torch.device(0))
                        action_share[i] = torch.tensor(action, device=torch.device(0))
                        reward_share[i] = torch.tensor(reward, device=torch.device(0))
                        next_state_share[i] = torch.tensor(next_state, device=torch.device(0))
                        done_share[i] = torch.tensor(done, device=torch.device(0))
                    memory_share_list = [state_share, action_share, reward_share, next_state_share, done_share]
                    self.__worker_pipe.send([True, memory_share_list]) # the first one denoteing construction of share memory
                else:
                    self.__worker_pipe.send([False, self.out_pointer])
                    self.current_cache_size-=1
                    self.out_pointer = (self.out_pointer + 1)%self.cache_size

                    state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                    state_share[self.in_pointer] = torch.tensor(state, device=torch.device(0))
                    action_share[self.in_pointer] = torch.tensor(action, device=torch.device(0))
                    reward_share[self.in_pointer] = torch.tensor(reward, device=torch.device(0))
                    next_state_share[self.in_pointer] = torch.tensor(next_state, device=torch.device(0))
                    done_share[self.in_pointer] = torch.tensor(done, device=torch.device(0))
                    self.in_pointer = (self.in_pointer + 1) % self.cache_size
                    self.current_cache_size+=1

            elif op == self.CLOSE:
                self.__worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.__pipe.send([self.ADD, data])

    def sample(self):
        self.__pipe.send([self.SAMPLE, None])
        is_construct_cache, data = self.__pipe.recv()
        if is_construct_cache:
            self.state_share, self.action_share, self.reward_share, self.next_state_share, self.done_share =  data
            return self.state_share[0], self.action_share[0], self.reward_share[0], self.next_state_share[0], self.done_share[0]
        else:
            return self.state_share[data], self.action_share[data], self.reward_share[data], self.next_state_share[data], self.done_share[data]

    def close(self):
        self.__pipe.send([self.CLOSE, None])
        self.__pipe.close()
