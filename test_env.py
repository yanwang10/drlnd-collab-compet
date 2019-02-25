import gym
import json
import numpy as np
from src import *
from unityagents import UnityEnvironment
import numpy as np

def print_state(state):
    print('Agent 1 obs:', np.squeeze(state[0, :]))
    print('Agent 2 obs:', np.squeeze(state[1, :]))

np.set_printoptions(precision=2, linewidth=200)

env = UnityEnvironment(file_name="data/Tennis_Linux_NoVis/Tennis")
env = EnvWrapper(unity_env=env, brain_name='TennisBrain')

state = env.reset()

print_state(state)

while True:
    raw_action = input("Act> ")
    if raw_action == 'quit':
        break
    actions = list(map(float, raw_action.split(' ')))
    if len(actions) != 4:
        print("Please input 4 numbers separated by spaces as action")
        continue
    actions = np.array(actions)
    ns, r, d = env.step(actions)
    print('rewards:', r, 'dones:', d)
    print_state(ns)
    