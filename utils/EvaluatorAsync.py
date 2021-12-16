import torch
import numpy as np
from numpy import random
import torch.multiprocessing as mp
from statistics import mean
from utils.LogAsync import logger
import time
from collections import deque
import os
from datetime import datetime
import imageio
import matplotlib
from .ActorAsync import ActorAsync

class EvaluatorAsync(mp.Process):
    EVAL = 0
    # NETWORK = 1
    UPDATE_POLICY=1
    SAVE_POLICY=2
    EXIT = 3

    def __init__(self, eval_actor_class:ActorAsync, env, seed=None, *arg, **args):
        mp.Process.__init__(self)
        self.args = args
        self._pipe, self._worker_pipe = mp.Pipe()
        self.seed = random.randint(0,10000) if seed == None else seed
        self.eval_actor_class = eval_actor_class
        self.env = env

    def run(self):
        self.eval_render_save_video = None if self.args['eval_render_save_video'] is None else [int(i) for i in self.args['eval_render_save_video']]
        self.eval_actor = self.eval_actor_class(env = self.env, network_lock=mp.Lock())
        self.eval_actor.start()
        if not self.args['eval_display']: matplotlib.use('Agg')

        while True:
            (cmd, msg) = self._worker_pipe.recv()
            if cmd == self.EVAL:
                self._eval(*(msg[0]), **(msg[1]))
                
            elif cmd == self.UPDATE_POLICY:
                self.eval_actor.update_policy(*(msg[0]), **(msg[1])) #TODO 拿出来

            elif cmd == self.SAVE_POLICY:
                self.eval_actor.save_policy(*(msg[0]), **(msg[1])) #TODO 拿出来

            elif cmd == self.EXIT:
                self._worker_pipe.close()
                return 

            # elif cmd == self.NETWORK:
            #     self.evaluator_policy = msg
            #     now = datetime.now()
            #     self.evaluator_name = self.evaluator_policy.__class__.__name__ + '(' + self.args['env_name'] + ')_%d_'%self.args['seed'] + now.strftime("%Y%m%d-%H%M%S")
            #     self.gif_folder = 'save_video/' + self.evaluator_name + '/'
            #     if not os.path.exists(self.gif_folder):
            #         os.makedirs(self.gif_folder)
            #     if not os.path.exists('save_model'):
            #         os.makedirs('save_model')

            else:
                raise NotImplementedError

    # def save_policy(self):
    #     #TODO
    #     if eval_idx == 1 or ep_rewards_list_mean >= best_ep_rewards_list_mean:
    #         torch.save(self.evaluator_policy.state_dict(), 'save_model/' + self.evaluator_name + '.pt')
    #         best_ep_rewards_list_mean = ep_rewards_list_mean
    #         logger.add({'eval_best': best_ep_rewards_list_mean})
        

    def eval(self, eval_idx = 0, *arg, **args):
        # #TODO
        args['eval_idx'] = eval_idx
        # with self.evaluator_lock:
        #     # 执行完上一次Eval才能进行下一次
        #     if self.args['mode'] == 'eval': # if this is only an evaluation session, then load model first
        #         if self.args['model_path'] is None: raise Exception("Model Path for Evaluation is not given! Include --model_path")
        #         self.evaluator_policy.load_state_dict(torch.load(self.args['model_path']))
        #     else:
        #         self.evaluator_policy.load_state_dict(state_dict)
        
        self._pipe.send([self.EVAL, (arg, args)])
        