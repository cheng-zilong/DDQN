import torch
import numpy as np
import torch.multiprocessing as mp
import random 

class ActorAsync(mp.Process):
    STEP = 0
    EXIT = 1
    NETWORK = 2
    def __init__(self, env, step_method, *arg, **args):
        mp.Process.__init__(self)
        self.seed = args['seed']
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.env = env
        self.steps_no = args['train_freq']
        self.step_method = step_method
        self.start()
    
    def reset(self):
        return self.env.reset()

    def init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.np_random.seed(self.seed)

    def collect(self, *arg, **args):
        data = []
        for _ in range(self.steps_no):
            one_step = self.step_method(self.env, *arg, **args)
            data.append(one_step) 
        return data

    def run(self):
        self.init_seed()
        is_init_cache = False
        while True:
            (cmd, msg) = self.__worker_pipe.recv()
            if cmd == self.STEP:
                if not is_init_cache:
                    is_init_cache = True
                    self.__worker_pipe.send(self.collect(*(msg[0]), **(msg[1])))
                    self.cache = self.collect(*(msg[0]), **(msg[1]))
                else:
                    self.__worker_pipe.send(self.cache)
                    self.cache = self.collect(*(msg[0]), **(msg[1]))

            elif cmd == self.EXIT:
                self.__worker_pipe.close()
                return

            elif cmd == self.NETWORK:
                self._network = msg

            else:
                raise NotImplementedError

    def step(self, *arg, **args):
        self.__pipe.send([self.STEP, (arg, args)])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])