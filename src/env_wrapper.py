import numpy as np
import time

UNITY = 'unity'

class EnvWrapper:
    """
    The wrapper that wraps environment from unity ml-agent.
    """
    def __init__(self, unity_env=None, brain_name=None, max_episode_len=1000):
        self.max_episode_len = max_episode_len
        self.this_episode_len = 0
        self.env = None
        self.cur_state = None
        if unity_env:
            print('Using unity env with brain name "%s"' % brain_name)
            self.env = unity_env
            self.brain_name = brain_name
            self.env_type = UNITY
        else:
            raise NotImplementedError
    
    def get_current_state(self):
        return self.cur_state

    def transform_obs(self, obs):
        if self.env_type == UNITY:
            # if len(obs.shape) == 1:
            #     return np.expand_dims(obs, axis=0)
            # obs = obs.reshape(1, -1)
            return obs
        else:
            raise NotImplementedError

    def reset(self, train_mode=True):
        self.this_episode_len = 0
        if self.env_type == UNITY:
            env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
            self.cur_state = self.transform_obs(env_info.vector_observations)
        else:
            raise NotImplementedError
        return self.get_current_state()

    def step(self, action):
        self.this_episode_len += 1
        # time.sleep(1)
        if self.env_type == UNITY:
            env_info = self.env.step(action)[self.brain_name]
            obs = env_info.vector_observations
            reward = np.array(env_info.rewards)
            done = env_info.local_done
            self.cur_state = self.transform_obs(obs)
            # for i in range(len(done)):
            #     done[i] |= self.this_episode_len >= self.max_episode_len
            return self.cur_state, reward, done
        else:
            raise NotImplementedError
