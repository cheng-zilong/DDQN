import gym
import numpy as np
from gym.spaces.box import Box
from collections import deque
from gym import spaces
from baselines.common.atari_wrappers import FrameStack as FrameStack_, make_atari, wrap_deepmind

def make_atari_env(seed, env_name, max_episode_steps, 
    episode_life, clip_reward, stack_frames, **kwargs):
        env = make_atari(env_name, max_episode_steps=max_episode_steps)
        env = TotalRewardWrapper(env)
        env = wrap_deepmind(env,
                            episode_life=episode_life,
                            clip_rewards=clip_reward,
                            frame_stack=False,
                            scale=False)
        env = TransposeImage(env)
        env = FrameStack(env = env, k = stack_frames)
        env.seed(seed)
        env.action_space.np_random.seed(seed)
        return env

class TotalRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.float32(reward)
        self.total_rewards += reward
        info['total_rewards'] = self.total_rewards
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info


class TransposeImage(gym.ObservationWrapper):
    '''
    Swap the index to fit pytorch
    '''
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


'''
origianl one stack at the last index, we hope to stack at the first index
'''
class FrameStack(FrameStack_):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=env.observation_space.dtype)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]