import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99, help="The parameter gamma in value iteration. Discount parameter of the future reward.")
    parser.add_argument('--lr', type=float, default=0.00025, help="Learning rate in the neural netowrk optimizer.")
    parser.add_argument('--optimizer_eps', type=float, default=0.01 / 32, help="The parameter eps in the Adam optimizer.")
    parser.add_argument('--clip_gradient', type = float, default = 0.5, help="Gradient clip in the neural network training process.")
    parser.add_argument('--clip_reward', type = bool, default = True, help="Reward clip as is proposed by Nature DQN paper for Atari.")
    parser.add_argument('--episode_life', type = bool, default = True, help="Episode life as is given in the Atari wrapper.")
    parser.add_argument('--max_episode_steps', type = int, default = None, help="Maximum episode steps for the Atari wrapper.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size for training.")
    parser.add_argument('--seed', type=int, default=4, help="Seed for the platform.")
    parser.add_argument('--eps_start', type=int, default=1, help="Linear decay for epsilon-greedy, the start epsilon.")
    parser.add_argument('--eps_end', type=int, default=0.01, help="Linear decay for epsilon-greedy, the end epsilon.")
    parser.add_argument('--eps_decay_steps', type=int, default=int(1e6), help="Linear decay for epsilon-greedy, how many steps from start to end.")
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help="The size of the training dataset.")
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', help="Gym environment name.")
    parser.add_argument('--stack_frames', type=int, default=4, help="The number of frames stacked and used for prediction.")
    parser.add_argument('--sim_steps', type=int, default=int(5e7))
    parser.add_argument('--train_steps', type=int, default=int(1e7))
    parser.add_argument('--train_start_step', type=int, default=50000)
    parser.add_argument('--train_network_freq', type=int, default=4)
    parser.add_argument('--train_update_target_freq', type=int, default=40000)
    parser.add_argument('--mode', type=str, default='train') # eval
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--ep_reward_avg_number', type=int, default = 10)
    parser.add_argument('--project_name', type=str, default = 'C51')
    
    # Evaluation
    parser.add_argument('--eval_max_steps', type=int, default=18000, help="The maximum steps for each episode in evaluation.")
    parser.add_argument('--eval_freq', type=int, default=int(1e6), help="Every *eval_freq* training steps, Evaluate the model.")
    parser.add_argument('--eval_display', type=bool, default=False, help="Whether the evaluation procedure is displayed.")
    parser.add_argument('--eval_number', type=int, default=30, help="In each evaluation, *eval_number* episodes will be implemented.")
    parser.add_argument('--eval_eps', type=float, default=0.001, help="*eval_eps* probability to choose a random action in evaluation.")
    parser.add_argument('--eval_video_fps', type=int, default=60/4, help="Export *eval_video_fps* frames per second in the video")

    # C51
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=-10.)
    parser.add_argument('--v_max', type=float, default=10.)
    return parser
