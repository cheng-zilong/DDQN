from cv2 import data
import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import random
from wrapper import LazyFrames

from baselines.deepq.replay_buffer import ReplayBuffer

class ReplayBufferAsync(mp.Process):
    '''
    add numpy
    sample torch.tensor.cuda()
    '''
    ADD = 0
    SAMPLE = 1
    CLOSE = 2

    def __init__(self, *arg, **args):
        mp.Process.__init__(self)
        self.buffer_size = args['buffer_size']
        self.batch_size = args['batch_size']
        self.stack_frames = args['stack_frames']
        self.cache_size = 2
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.is_init_cache = False
        self.out_pointer = 1 # output pointer 0 when initialize, 1 when first output
        self.in_pointer = 0 # update pointer 0 when first update
        self.seed = args['seed']
        self.last_frames = None 
        self.start()

    def init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        self.init_seed()
        replay_buffer = ReplayBuffer(self.buffer_size)
        memory_share_list = []
        frames = deque([], maxlen=self.stack_frames)
        while True:
            cmd, data = self.__worker_pipe.recv()
            if cmd == self.ADD:
                action, obs, reward, done = data
                if action is None: #if reset
                    for _ in range(self.stack_frames):
                        frames.append(obs)
                    self.last_frames = LazyFrames(list(frames))
                else:
                    frames.append(obs)
                    replay_buffer.add(self.last_frames, action, reward, LazyFrames(list(frames)), done)
                    self.last_frames = LazyFrames(list(frames))

            elif cmd == self.SAMPLE:
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
                    self.out_pointer = (self.out_pointer + 1)%self.cache_size

                    state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
                    state_share[self.in_pointer] = torch.tensor(state, device=torch.device(0))
                    action_share[self.in_pointer] = torch.tensor(action, device=torch.device(0))
                    reward_share[self.in_pointer] = torch.tensor(reward, device=torch.device(0))
                    next_state_share[self.in_pointer] = torch.tensor(next_state, device=torch.device(0))
                    done_share[self.in_pointer] = torch.tensor(done, device=torch.device(0))
                    self.in_pointer = (self.in_pointer + 1) % self.cache_size

            elif cmd == self.CLOSE:
                self.__worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def add(self, action, obs, reward, done):
        '''
        if action is none, it is the reset frame
        '''
        data = (action, obs, reward, done)
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
