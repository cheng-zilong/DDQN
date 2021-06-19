import torch
import numpy as np
import torch.multiprocessing as mp
from collections import deque
import random 

class ActorAsync(mp.Process):
    STEP = 0
    EXIT = 1
    NETWORK = 2
    def __init__(self, env, steps_no, seed, lock):
        mp.Process.__init__(self)
        self.seed = seed
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.env = env
        self.is_init_cache = False
        self.cache_size = 2
        self.eps = 1
        self.lock = lock
        self.steps_no = steps_no
        self.start()

    def run(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        cache = deque([], maxlen=self.cache_size)

        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                eps = data
                if not self.is_init_cache:
                    self.is_init_cache = True
                    self.state = self.env.reset()
                    self.__worker_pipe.send(self.eps_greedy_step(eps))
                    for _ in range(self.cache_size):
                        cache.append(self.eps_greedy_step(eps))
                else:
                    self.__worker_pipe.send(cache.popleft())
                    cache.append(self.eps_greedy_step(eps))

            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def eps_greedy_step(self, eps):
        # auto reset
        data = []
        for _ in range(self.steps_no):
            eps_prob =  random.random()
            if eps_prob > eps:
                with self.lock:
                    action = self._network.act(self.state)
            else:
                action = self.env.action_space.sample()

            next_state, reward, done, info = self.env.step(action)
            data.append([self.state, action, reward, next_state, done, info])
            if done:
                self.state = self.env.reset()
            else:
                self.state = next_state
        return data

    def step(self, eps):
        self.__pipe.send([self.STEP, eps])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])