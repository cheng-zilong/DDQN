import multiprocessing as mp
import wandb
from datetime import datetime

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class LogProcess(mp.Process, metaclass=Singleton):
    ADD = 0
    DELETE = 1
    WANDB_PRINT = 2
    TERMINAL_PRINT = 3
    EXIT = 4
    RENDER_FRAME = 5

    def __init__(self):
        mp.Process.__init__(self)
        
    def init(self, project_name, policy_class, env_name, seed, *args, **kwargs):
        self._project_name = project_name
        self._policy_name = policy_class
        self._env_name = env_name
        self._seed = seed
        self._args = args
        self._kwargs = kwargs
        self.__queue = mp.Queue(maxsize=50)
        self._log_dict = dict()
        now = datetime.now()
        self._run_name = self._policy_name + '(' + self._env_name + ')_%d_'%self._seed + now.strftime("%Y%m%d-%H%M%S")
        self.start()

    @property
    def run_name(self):
        return self._run_name

    @property
    def project_name(self):
        return self._project_name

    def run(self):
        wandb.init(name=self._run_name, project=self._project_name, config=self._kwargs)
        while True:
            cmd, data = self.__queue.get()
            if cmd == self.ADD:
                for key in data:
                    self._log_dict[key] = data[key]
                
            elif cmd == self.DELETE:
                if isinstance(data, (list,tuple)):
                    for key in data:
                        self._log_dict.pop(key, None)
                else:
                    self._log_dict.pop(data, None)

            elif cmd == self.WANDB_PRINT:
                caption, step = data
                wandb.log(self._log_dict, step=step)
                print(caption)
                print('log_idx:  %d'%(step))
                for key in self._log_dict:
                    if not isinstance(self._log_dict[key], (wandb.Image, wandb.Video)): 
                        print(key +':  ' + str(self._log_dict[key]))
                print('-------------------')

            elif cmd == self.TERMINAL_PRINT:
                caption, log_dict_tmp = data 
                if caption != "":
                    print(caption) 
                if log_dict_tmp != None:
                    for key in log_dict_tmp:
                        print(key +':  ' + str(log_dict_tmp[key]))
                print('-------------------')
                
            elif cmd == self.EXIT:
                return 

            else:
                raise NotImplementedError

    def add(self, data):
        self.__queue.put([self.ADD, data])
        
    def delete(self, keys):
        self.__queue.put([self.DELETE, keys])

    def wandb_print(self, caption, step):
        self.__queue.put([self.WANDB_PRINT, [caption, step]])

    def terminal_print(self, caption = "", log_dict_tmp = None):
        self.__queue.put([self.TERMINAL_PRINT, [caption, log_dict_tmp]])

    def exit(self):
        self.__queue.put([self.EXIT, None])
        self.__queue.close()

logger = LogProcess()