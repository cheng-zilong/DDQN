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

    def __init__(self, buffer_size, batch_size, stack_frames, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self._replay_buffer = ReplayBuffer(self.buffer_size)
        self._frames = deque([], maxlen=self.stack_frames)
        self._is_init_cache = False
        while True:
            cmd, msg = self._receive()
            if cmd == self.ADD:
                self._add(*msg)
                
            elif cmd == self.SAMPLE:
                self._sample(msg)

            elif cmd == self.CLOSE:
                self._worker_pipe.close()
                return
                
            else:
                raise Exception('Unknown command')

    def add(self, action, obs, reward, done):
        '''
        如果stack_frame != 1，就假设数据已经堆叠好了,所以取最后一个数据
        如果stack_frame == 1，就假设数据是单个产生的,没有堆叠维度
        The format of data state must be
        [time stack dim, data dim] if it is lazyframes format
        if action is none, it is the reset frame
        '''
        obs = np.asarray(obs[None, -1]) if self.stack_frames!=1 else np.asarray(obs)
        data = (action, obs, reward, done)  
        if (not hasattr(self, '_data_example')) and data[0] is not None: #如果action不为None，那可以作为data example
            self._data_example = data
        self.send(self.ADD, data)

    def sample(self):
        ## return share tensor the first time, the return idx
        if not self.is_init_cache:
            self.is_init_cache=True
            if self.stack_frames == 1:
                self.state_share = torch.tensor([[self._data_example[1]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.action_share = torch.tensor([[self._data_example[0]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.reward_share = torch.tensor([[self._data_example[2]]*self.batch_size]*2, device=torch.device(0), dtype=torch.float32).share_memory_()
                self.next_state_share = torch.tensor([[self._data_example[1]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.done_share = torch.tensor([[self._data_example[3]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
            else:
                self.state_share = torch.tensor([[[self._data_example[1][0]]*self.stack_frames]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.action_share = torch.tensor([[self._data_example[0]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.reward_share = torch.tensor([[self._data_example[2]]*self.batch_size]*2, device=torch.device(0), dtype=torch.float32).share_memory_()
                self.next_state_share = torch.tensor([[[self._data_example[1][0]]*self.stack_frames]*self.batch_size]*2, device=torch.device(0)).share_memory_()
                self.done_share = torch.tensor([[self._data_example[3]]*self.batch_size]*2, device=torch.device(0)).share_memory_()
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

    def _sample(self, msg):
        if not self._is_init_cache:
            self._is_init_cache=True
            self._state_share, self._action_share, self._reward_share, self._next_state_share, self._done_share = msg
            for i in range(0, 2): # no need update 0
                state, action, reward, next_state, done = self._replay_buffer.sample(self.batch_size)
                self._state_share[i] = torch.tensor(state, device=torch.device(0))
                self._action_share[i] = torch.tensor(action, device=torch.device(0))
                self._reward_share[i] = torch.tensor(reward, device=torch.device(0))
                self._next_state_share[i] = torch.tensor(next_state, device=torch.device(0))
                self._done_share[i] = torch.tensor(done, device=torch.device(0))
            self._send(True)
        else:
            self._send(True)
            state, action, reward, next_state, done = self._replay_buffer.sample(self.batch_size)
            self._state_share[self.in_pointer] = torch.tensor(state, device=torch.device(0))
            self._action_share[self.in_pointer] = torch.tensor(action, device=torch.device(0))
            self._reward_share[self.in_pointer] = torch.tensor(reward, device=torch.device(0))
            self._next_state_share[self.in_pointer] = torch.tensor(next_state, device=torch.device(0))
            self._done_share[self.in_pointer] = torch.tensor(done, device=torch.device(0))
            self.in_pointer = 0 if self.in_pointer == 1 else 1

    def _add(self, action, obs, reward, done):
        if action is None: #if reset
            if self.stack_frames == 1:
                self._last_frames = obs
            else:
                for _ in range(self.stack_frames):
                    self._frames.append(obs)
                self._last_frames = LazyFrames(list(self._frames))
        else:
            if self.stack_frames == 1:
                self._replay_buffer.add(np.array(self._last_frames,copy=True), 
                                        np.array(action,copy=True), 
                                        np.array(reward, dtype=np.float32,copy=True), 
                                        np.array(obs,copy=True), 
                                        np.array(done,copy=True))
                self._last_frames = obs
            else:
                self._frames.append(obs)
                current_frames = LazyFrames(list(self._frames))
                self._replay_buffer.add(self._last_frames, action, reward, current_frames, done)
                self._last_frames = current_frames