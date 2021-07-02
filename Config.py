import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--opt_eps', type=float, default=0.01 / 32)
    parser.add_argument('--gradient_clip', type = float, default = 0.5)
    parser.add_argument('--clip_reward', type = bool, default = True)
    parser.add_argument('--episode_life', type = bool, default = True)
    parser.add_argument('--max_episode_steps', type = int, default = None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--eps_start', type=int, default=1)
    # parser.add_argument('--eps_end', type=int, default=0.01)
    parser.add_argument('--eps_end', type=int, default=0.1)
    parser.add_argument('--eps_decay_steps', type=int, default=int(1e6))
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--stack_frames', type=int, default=4)
    parser.add_argument('--train_steps', type=int, default=int(1e7))
    parser.add_argument('--start_training_steps', type=int, default=50000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--update_target_steps', type=int, default=40000)
    parser.add_argument('--mode', type=str, default='train') # eval
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--ep_reward_avg_number', type=int, default = 10)
    

    parser.add_argument('--eval_steps', type=int, default=18000)
    parser.add_argument('--eval_freq', type=int, default=int(1e6))
    parser.add_argument('--eval_display', type=bool, default=False)
    parser.add_argument('--eval_number', type=int, default=30)
    parser.add_argument('--eval_render_freq', type=int, default=5)
    parser.add_argument('--eval_eps', type=float, default=0.05)
    parser.add_argument('--eval_render_save_gif', type=tuple, default=None) # if None, then save all episode in gif, otherwise [1, 5] means only save 1 and 5 episode

    return parser
