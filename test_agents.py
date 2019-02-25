import gym
import json
import numpy as np
from src import *
from unityagents import UnityEnvironment
import numpy as np

np.set_printoptions(precision=3)

env = UnityEnvironment(file_name="data/Tennis_Linux_NoVis/Tennis")
env = EnvWrapper(unity_env=env, brain_name='TennisBrain')

config = {
    # Configs about the env.
    'agent_num': 2,
    'state_size': 24,
    'action_size': 2,
    'out_low': -1,
    'out_high': 1,
    
    # Configs for the individual DDPG agent.
    'tau': 1e-3,
    'gamma': 0.95,
    'init_weight_scale': 1.,
    'grad_clip': 10.,
    'actor_hidden': [256, 128, 64],
    'actor_lr': 1e-3,
    'critic_hidden': [512, 256, 128],
    'critic_lr': 1e-4,
    'action_repeat': 1,
    
    # Configs for the training process.
    'noise_discount': 0.9999,
    'seed': 1317317,
    'buffer_size': int(1e7),
    'batch_num': 1024,
    'model_dir': './saved_model',
    'max_episode_num': 1e6,
    'max_step_num': 1e8,
    'learn_interval': 100,
    
    # Configs for logging.
    'log_file': './log.pickle',
    'window_size': 100,
    'log_interval': 100,
    'tensorboard_log_dir': './tblog',
}
# print(json.dumps(config, indent=4))
agents = MADDPGAgents(config)
TrainMADDPG(env, agents, config)
