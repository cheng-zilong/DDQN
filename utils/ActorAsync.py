import torch
import numpy as np
import torch.multiprocessing as mp
import random 
import time
from statistics import mean
from collections import deque
from utils.LogAsync import logger
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np 
class ActorAsync(mp.Process):
    STEP = 0
    EXIT = 1
    RESET = 2
    UPDATE_POLICY=3
    SAVE_POLICY=4
    EVAL=5
    def __init__(self, env, seed = None, *arg, **args):
        mp.Process.__init__(self)
        self._pipe, self._worker_pipe = mp.Pipe()
        self.env = env
        self.seed = random.randint(0,10000) if seed == None else seed
        self.arg = arg 
        self.args = args 


    def run(self):
        self._init_seed()
        is_init_cache = False
        while True:
            (cmd, msg) = self._worker_pipe.recv()
            if cmd == self.STEP:
                if not is_init_cache:
                    is_init_cache = True
                    self._worker_pipe.send(self._collect(*(msg[0]), **(msg[1])))
                    self.cache = self._collect(*(msg[0]), **(msg[1]))
                else:
                    self._worker_pipe.send(self.cache)
                    self.cache = self._collect(*(msg[0]), **(msg[1]))

            elif cmd == self.RESET:
                self._worker_pipe.send(self._reset())
                is_init_cache = False

            elif cmd == self.EVAL:
                self._eval(*(msg[0]), **(msg[1]))

            elif cmd == self.UPDATE_POLICY:
                self._update_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.SAVE_POLICY:
                self._save_policy(*(msg[0]), **(msg[1]))

            elif cmd == self.EXIT:
                self._worker_pipe.close()
                return

            else:
                raise NotImplementedError

    def collect(self, steps_no, *arg, **args):
        args['steps_no'] = steps_no
        self._pipe.send([self.STEP, (arg, args)])
        return self._pipe.recv()

    def reset(self):
        self._pipe.send([self.RESET, None])
        return self._pipe.recv()

    def close(self):
        self._pipe.send([self.EXIT, None])
        self._pipe.close()

    def update_policy(self, *arg, **args):
        self._pipe.send([self.UPDATE_POLICY, (arg, args)])

    def save_policy(self, *arg, **args):
        self._pipe.send([self.SAVE_POLICY, (arg, args)])

    def eval(self, eval_idx = 0, *arg, **args):
        # #TODO
        args['eval_idx'] = eval_idx
        # with self.evaluator_lock:
        #     if self.args['mode'] == 'eval': # if this is only an evaluation session, then load model first
        #         if self.args['model_path'] is None: raise Exception("Model Path for Evaluation is not given! Include --model_path")
        #         self.evaluator_policy.load_state_dict(torch.load(self.args['model_path']))
        #     else:
        #         self.evaluator_policy.load_state_dict(state_dict)
        
        self._pipe.send([self.EVAL, (arg, args)])

    def _init_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.np_random.seed(self.seed)

    def _reset(self):
        self._actor_last_state = self.env.reset()
        self._actor_done_flag = False
        return self._actor_last_state

    def _collect(self, *arg, **args):
        data = []
        for _ in range(args['steps_no']):
            one_step = self._step(*arg, **args)
            data.append(one_step) 
        return data

    def _eval_ep(self, ep_idx, *arg, **args):
        # Evaluate one episode
        state = self._reset()
        tic   = time.time()
        for ep_steps_idx in range(1, self.args['eval_max_steps'] + 1):
            action, state, reward, done, info = self._collect(steps_no = 1, *arg, **args)[-1] 
            # step returns a list [[action, state, reward, done, info]]
            # if (ep_idx is None or ep_idx in self.eval_render_save_video) and \
            #     (ep_steps_idx-1) % self.args['eval_render_freq'] == 0 : # every eval_render_freq frames sample 1 frame
            #     self.evaluator_policy._render_frame(self.eval_actor, state, action, self.writer) TODO
            if done:
                if info['episodic_return'] is not None: break
        toc = time.time()
        fps = ep_steps_idx / (toc-tic)
        return ep_steps_idx, info['total_rewards'], fps

    def _eval(self, *arg, **args):
        eval_idx = args['eval_idx']
        ep_rewards_list = deque(maxlen=self.args['eval_number'])
        for ep_idx in range(1, self.args['eval_number']+1):
            # self.writer = imageio.get_writer(self.gif_folder + '%08d_%03d.mp4'%(eval_idx, ep_idx), fps = self.args['eval_video_fps'])#TODO
            ep_steps, ep_rewards, fps = self._eval_ep(ep_idx, *arg, **args)
            # self.writer.close()#TODO
            ep_rewards_list.append(ep_rewards)
            ep_rewards_list_mean = mean(ep_rewards_list)
            logger.terminal_print(
                '--------(Evaluating Agent: %d)'%(eval_idx), {
                '--------ep': ep_idx, 
                '--------ep_steps':  ep_steps, 
                '--------ep_reward': ep_rewards, 
                '--------ep_reward_mean': ep_rewards_list_mean, 
                '--------fps': fps})
        logger.add({'eval_last': ep_rewards_list_mean})

    def _step(self, *arg, **args):
        '''
        This is a template for the _step method
        Overwrite this method to implement your own _step method
        return [[action, state, reward, done, info], ...]
        '''
        pass 

    def _update_policy(self, *arg, **args):
        '''
        This is only a template for the _update_policy method
        Overwrite this method to implement your own _update_policy method
        '''
        pass

    def _save_policy(self, *arg, **args):
        '''
        This is only a template for the _save_policy method
        Overwrite this method to implement your own _save_policy method
        '''
        pass

    #TODO
    def _render_frame(self, actor, state, action, writer):
        if self.my_fig is None:
            self.my_fig = plt.figure(figsize=(10, 5), dpi=160)
            plt.rcParams['font.size'] = '8'
            gs = gridspec.GridSpec(1, 2)
            self.ax_left = self.my_fig.add_subplot(gs[0])
            self.ax_right = self.my_fig.add_subplot(gs[1])
            self.my_fig.tight_layout()
            self.fig_pixel_cols, self.fig_pixel_rows = self.my_fig.canvas.get_width_height()
        action_prob = np.swapaxes(self.action_prob[0].cpu().numpy(),0, 1)
        legends = []
        for i, action_meaning in enumerate(actor.env.unwrapped.get_action_meanings()):
            legend_text = ' (Q=%+.2e)'%(self.action_Q[0,i]) if i == action else ' (Q=%+.2e)*'%(self.action_Q[0,i])
            legends.append(action_meaning + legend_text) 
        self.ax_left.clear()
        self.ax_left.imshow(state[-1])
        self.ax_left.axis('off')
        self.ax_right.clear()
        self.ax_right.plot(self.atoms_cpu, action_prob)
        self.ax_right.legend(legends)
        self.ax_right.grid(True)
        self.my_fig.canvas.draw()
        buf = self.my_fig.canvas.tostring_rgb()
        writer.append_data(np.fromstring(buf, dtype=np.uint8).reshape(self.fig_pixel_rows, self.fig_pixel_cols, 3))