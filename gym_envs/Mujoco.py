import gym
from gym_envs.AtariWrapper import TotalRewardWrapper
def make_mujoco(env_name, *args, **kwargs):
    env = gym.make(env_name)
    env = TotalRewardWrapper(env)
    return env
