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
        self.eps = 1
        self.lock = lock
        self.steps_no = steps_no
        self.done = True
        self.start()

    def run(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                eps = data
                if not self.is_init_cache:
                    self.is_init_cache = True
                    self.__worker_pipe.send(self.eps_greedy_step(eps))
                    self.cache = self.eps_greedy_step(eps)
                else:
                    self.__worker_pipe.send(self.cache)
                    self.cache = self.eps_greedy_step(eps)

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
            if self.done:
                self.state = self.env.reset()
                data.append([None, self.state, None, None, None])
                self.done = False
                continue
            eps_prob =  random.random()
            if eps_prob > eps:
                with self.lock:
                    action = self._network.act(np.array(self.state, copy=False))
            else:
                action = self.env.action_space.sample()

            obs, reward, self.done, info = self.env.step(action)
            data.append([action, obs, reward, self.done, info])
            self.state = obs
        return data

    def step(self, eps):
        self.__pipe.send([self.STEP, eps])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])