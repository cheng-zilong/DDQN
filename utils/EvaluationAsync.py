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

class EvaluationAsync(mp.Process):
    EVAL = 0
    NETWORK = 1
    EXIT = 3

    def __init__(self, make_env_fun, **args):
        mp.Process.__init__(self)
        self.make_env_fun = make_env_fun
        self.args = args
        self.__pipe, self.__worker_pipe = mp.Pipe()
        self.evaluator_lock = mp.Lock()
        self.seed = args['seed']
        self.start()

    def _eval(self, ep_idx):
        env = self.make_env_fun(**self.args)
        env.seed(self.seed+ep_idx)
        env.action_space.np_random.seed(self.seed+ep_idx)
        state = env.reset()
        tic   = time.time()
        for eval_steps_idx in range(1, self.eval_steps + 1):
            eps_prob =  random.random()
            action = self.evaluator_network.act(state) if eps_prob > self.eval_eps else env.action_space.sample()
            state, _, done, info = env.step(action)
            if (ep_idx is None or ep_idx in self.eval_render_save_video) and \
                (eval_steps_idx-1) % self.eval_render_freq == 0 : # every eval_render_freq frames sample 1 frame
                self.evaluator_network._render_frame(env, state, action, self.writer)
            if done:
                state = env.reset()
                if info['episodic_return'] is not None: break
        toc = time.time()
        fps = eval_steps_idx / (toc-tic)
        return eval_steps_idx, info['total_rewards'], fps

    def init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        self.init_seed()
        self.eval_steps = self.args['eval_steps']
        self.eval_number = self.args['eval_number']
        self.eval_render_freq = self.args['eval_render_freq']
        self.eval_eps = self.args['eval_eps']
        self.eval_render_save_video = None if self.args['eval_render_save_video'] is None else [int(i) for i in self.args['eval_render_save_video']]
        if not self.args['eval_display']: matplotlib.use('Agg')
        video_fps = 60/4/self.args['eval_render_freq']

        while True:
            cmd, data = self.__worker_pipe.recv()
            if cmd == self.EVAL:
                with self.evaluator_lock:
                    self.current_train_steps = data
                    ep_rewards_list = deque(maxlen=self.eval_number)
                    for ep_idx in range(1, self.eval_number+1):
                        self.writer = imageio.get_writer(self.gif_folder + '%08d_%03d.mp4'%(self.current_train_steps, ep_idx), fps = video_fps)
                        eval_steps_idx, ep_rewards, fps = self._eval(ep_idx)
                        self.writer.close()
                        ep_rewards_list.append(ep_rewards)
                        ep_rewards_list_mean = mean(ep_rewards_list)
                        logger.terminal_print('--------(Evaluating Agent: %d)'%(self.current_train_steps), {
                            '--------ep': ep_idx, 
                            '--------ep_steps':  eval_steps_idx, 
                            '--------ep_reward': ep_rewards, 
                            '--------ep_reward_mean': ep_rewards_list_mean, 
                            '--------fps': fps})
                    logger.add({'eval_last': ep_rewards_list_mean})
                    if self.current_train_steps == 1 or ep_rewards_list_mean >= best_ep_rewards_list_mean:
                        torch.save(self.evaluator_network.state_dict(), 'save_model/' + self.evaluator_name + '.pt')
                        best_ep_rewards_list_mean = ep_rewards_list_mean
                        logger.add({'eval_best': best_ep_rewards_list_mean})

            elif cmd == self.EXIT:
                self.__worker_pipe.close()
                return 

            elif cmd == self.NETWORK:
                self.evaluator_network = data
                now = datetime.now()
                self.evaluator_name = self.evaluator_network.__class__.__name__ + '(' + self.args['env_name'] + ')_%d_'%self.args['seed'] + now.strftime("%Y%m%d-%H%M%S")
                self.gif_folder = 'save_video/' + self.evaluator_name + '/'
                if not os.path.exists(self.gif_folder):
                    os.makedirs(self.gif_folder)
                if not os.path.exists('save_model'):
                    os.makedirs('save_model')

            else:
                raise NotImplementedError

    def init(self, netowrk_fun): 
        temp_env = self.make_env_fun(**self.args)
        self.evaluator_network  = netowrk_fun(temp_env.observation_space.shape, temp_env.action_space.n, **self.args).cuda().share_memory()
        self.__pipe.send([self.NETWORK, self.evaluator_network]) # pass network to the evaluation process

    def eval(self, train_steps = 0, state_dict = None):
        with self.evaluator_lock:
            if self.args['mode'] == 'eval': # if this is only an evaluation session, then load model first
                if self.args['model_path'] is None: raise Exception("Model Path for Evaluation is not given! Include --model_path")
                self.evaluator_network.load_state_dict(torch.load(self.args['model_path']))
            else:
                self.evaluator_network.load_state_dict(state_dict)
            self.__pipe.send([self.EVAL, train_steps])

    def exit(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

