import torch
import numpy as np
import torch.multiprocessing as mp
import wandb
import json

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class LogAsync(mp.Process, metaclass=Singleton):
    ADD = 0
    DELETE = 1
    WANDB_PRINT = 2
    TERMINAL_PRINT = 3
    EXIT = 4
    RENDER_FRAME = 5

    def __init__(self):
        mp.Process.__init__(self)
        
    def init(self, project_name = None, args = None):
        self.project_name = project_name
        self. args = args
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.log_dict = dict()
        self.start()

    def run(self):
        if self.project_name is not None:
            wandb.init(name='CatCnnDQN(' + self.args.env_name + ')_' + str(self.args.seed), project=self.project_name, config=self.args)
            self.wandb_init = True
        else:
            self.wandb_init = False
        while True:
            cmd, data = self.__worker_pipe.recv()
            if cmd == self.ADD:
                for key in data:
                    self.log_dict[key] = data[key]
                
            elif cmd == self.DELETE:
                if isinstance(data, (list,tuple)):
                    for key in data:
                        self.log_dict.pop(key, None)
                else:
                    self.log_dict.pop(data, None)

            elif cmd == self.WANDB_PRINT:
                caption, step = data
                if self.wandb_init:
                    wandb.log(self.log_dict, step=step)
                print(caption)
                for key in self.log_dict:
                    if not isinstance(self.log_dict[key], (wandb.Image, wandb.Video)): 
                        print(key +':  ' + str(self.log_dict[key]))
                print('-------------------')

            elif cmd == self.TERMINAL_PRINT:
                caption, log_dict_tmp = data 
                print(caption)
                for key in log_dict_tmp:
                    print(key +':  ' + str(log_dict_tmp[key]))
                print('-------------------')
                
            elif cmd == self.EXIT:
                self.__worker_pipe.close()
                return 

            else:
                raise NotImplementedError

    def add(self, data):
        self.__pipe.send([self.ADD, data])
        
    def delete(self, keys):
        self.__pipe.send([self.DELETE, keys])

    def wandb_print(self, caption, step):
        self.__pipe.send([self.WANDB_PRINT, [caption, step]])

    def terminal_print(self, caption, log_dict_tmp):
        self.__pipe.send([self.TERMINAL_PRINT, [caption, log_dict_tmp]])

    def exit(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

logger = LogAsync()