import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99, help="The parameter gamma in value iteration. Discount parameter of the future reward.")
    parser.add_argument('--lr', type=float, default=0.00025, help="Learning rate in the neural netowrk optimizer.")
    parser.add_argument('--opt_eps', type=float, default=0.01 / 32, help="The parameter eps in the Adam optimizer.")
    parser.add_argument('--gradient_clip', type = float, default = 0.5, help="Gradient clip in the neural network training process.")
    parser.add_argument('--clip_reward', type = bool, default = True, help="Reward clip as is proposed by Nature DQN paper for Atari.")
    parser.add_argument('--episode_life', type = bool, default = True, help="Episode life as is given in the Atari wrapper.")
    parser.add_argument('--max_episode_steps', type = int, default = None, help="Maximum episode steps for the Atari wrapper.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size for training.")
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--eps_start', type=int, default=1)
    parser.add_argument('--eps_end', type=int, default=0.01)
    parser.add_argument('--eps_decay_steps', type=int, default=int(1e6))
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--stack_frames', type=int, default=4)
    parser.add_argument('--train_steps', type=int, default=int(5e7))
    parser.add_argument('--start_training_steps', type=int, default=50000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--update_target_steps', type=int, default=40000)
    parser.add_argument('--mode', type=str, default='train') # eval
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--ep_reward_avg_number', type=int, default = 10)
    
    # Evaluation
    parser.add_argument('--eval_steps', type=int, default=18000, help="The maximum steps for each episode in evaluation.")
    parser.add_argument('--eval_freq', type=int, default=int(1e6), help="Every *eval_freq* training steps, Evaluate the model.")
    parser.add_argument('--eval_display', type=bool, default=False, help="Whether the evaluation is displayed.")
    parser.add_argument('--eval_number', type=int, default=30, help="In each evaluation, *eval_number* episodes will be implemented.")
    parser.add_argument('--eval_eps', type=float, default=0.001, help="*eval_eps* probability to choose a random action in evaluation.")
    parser.add_argument('--eval_render_freq', type=int, default=1, help="Every *eval_render_freq* evaluation steps, render the frames.")
    parser.add_argument('--eval_render_save_video', type=tuple, default=None, help="If None, then save all episode in gif, otherwise, \"1 10\" means only 1st and 10th episodes are saved. Notably, saving gif can be very slow!") # 

    # C51
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.)
    parser.add_argument('--v_max', type=float, default=10.)
    return parser
