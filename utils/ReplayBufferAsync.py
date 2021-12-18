from cv2 import data
import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import random
from gym_envs.AtariWrapper import LazyFrames

from baselines.deepq.replay_buffer import ReplayBuffer
from .Async import Async

class ReplayBufferAsync(Async):
    '''
    add numpy
    sample torch.tensor.cuda()
    '''
    ADD = 0
    SAMPLE = 1
    CLOSE = 2

    def __init__(self, buffer_size, batch_size, stack_frames, seed, *arg, **args):
        super().__init__(*arg, **args)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.stack_frames = stack_frames
        self.seed = seed
        self.is_init_cache = False
        self.in_pointer = 0 # update pointer 0 when first update

    def init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        self.init_seed()
        replay_buffer = ReplayBuffer(self.buffer_size)
        frames = deque([], maxlen=self.stack_frames)
        is_init_cache = False
        while True:
            cmd, msg = self._receive()
            if cmd == self.ADD:
                action, obs, reward, done = msg
                if action is None: #if reset
                    for _ in range(self.stack_frames):
                        frames.append(obs)
                    self.last_frames = LazyFrames(list(frames))
                else:
                    frames.append(obs)
                    current_frames = LazyFrames(list(frames))
                    replay_buffer.add(self.last_frames, action, reward, current_frames, done)
                    self.last_frames = current_frames

            elif cmd == self.SAMPLE:
                if not is_init_cache:
                    is_init_cache=True
                    state_share, action_share, reward_share, next_state_share, done_share = msg
                    for i in range(0, 2): # no need update 0
                        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                        state_share[i] = torch.tensor(state, device=torch.device(0))
                        action_share[i] = torch.tensor(action, device=torch.device(0))
                        reward_share[i] = torch.tensor(reward, device=torch.device(0))
                        next_state_share[i] = torch.tensor(next_state, device=torch.device(0))
                        done_share[i] = torch.tensor(done, device=torch.device(0))
                    self._send(True)
                else:
                    self._send(True)
                    state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                    state_share[self.in_pointer] = torch.tensor(state, device=torch.device(0))
                    action_share[self.in_pointer] = torch.tensor(action, device=torch.device(0))
                    reward_share[self.in_pointer] = torch.tensor(reward, device=torch.device(0))
                    next_state_share[self.in_pointer] = torch.tensor(next_state, device=torch.device(0))
                    done_share[self.in_pointer] = torch.tensor(done, device=torch.device(0))
                    self.in_pointer = 0 if self.in_pointer == 1 else 1

            elif cmd == self.CLOSE:
                self._worker_pipe.close()
                return
                
            else:
                raise Exception('Unknown command')

    def add(self, action, obs, reward, done):
        '''
        if action is none, it is the reset frame
        '''
        if isinstance(obs, LazyFrames):
            data = (action, obs[None, -1], reward, done)  
        else:
            data = (action, [obs], reward, done)  #加一个维度因为lazyframe叠第0维
        self.__last_data = data
        self.send(self.ADD, data)

    def sample(self):
        ## return share tensor the first time, the return idx
        if not self.is_init_cache:
            self.is_init_cache=True
            self.state_share = torch.tensor([[LazyFrames([self.__last_data[1]]*self.stack_frames)]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            self.action_share = torch.tensor([[self.__last_data[0]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            self.reward_share = torch.tensor([[self.__last_data[2]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            self.next_state_share = torch.tensor([[LazyFrames([self.__last_data[1]]*self.stack_frames)]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            self.done_share = torch.tensor([[self.__last_data[3]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            self.send(self.SAMPLE, (self.state_share, self.action_share, self.reward_share, self.next_state_share, self.done_share))
            self.receive()
            self.sample_idx = 0
        else:
            self.sample_idx  = 0 if self.sample_idx == 1 else 1
            self.send(self.SAMPLE, None)
            self.receive()
        return self.state_share[self.sample_idx], self.action_share[self.sample_idx], self.reward_share[self.sample_idx], self.next_state_share[self.sample_idx], self.done_share[self.sample_idx]

    def close(self):
        self.send(self.CLOSE, None)
        self._pipe.close()