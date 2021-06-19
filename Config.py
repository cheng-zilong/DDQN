import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--opt_eps', type=float, default=0.01 / 32)
    parser.add_argument('--gradient_clip', type = float, default = 0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--eps_start', type=int, default=1)
    parser.add_argument('--eps_end', type=int, default=0.01)
    parser.add_argument('--eps_decay_steps', type=int, default=int(1e6))
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--start_training_steps', type=int, default=50000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--update_target_freq', type=int, default=10000)
    return parser
