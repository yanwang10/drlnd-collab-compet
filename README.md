# drlnd-collab-compet
This repository is for the "Collaboration and Competition" project for Udacity
Deep Reinforcement Learning Nanodegree.

## About the task

In this project the environment is a simplified version of
[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
from Unity ml-agent project. This is an episodic task in which two agents
control the rackets to bounce a ball. The rewards are granted as following:

* +0.1 when hitting the ball over the net, and additional -0.01 if the ball
goes out of bounds
* -0.01 if missing the ball (the ball hits the ground)

The final reward of an episode is the maximum of two agents. In this project, we
aim to get >0.5 average reward of 100 consecutive episodes.

## About this solution

The basic dependencies are listed in this
[document](https://github.com/udacity/deep-reinforcement-learning#dependencies)
from Udacity Nanodegree. To run this solution, you will need to install some
additional dependencies:

`
conda install tensorboard protobuf
pip install torchsummary tensorboardX
`

To run the script, simply execute `python test_agents.py`. This script will
start training two agents with multi-agent DDPG algorithm with pre-defined
model structure and hyper-parameters. The result and analysis is in
[Report.ipynb](./Report.ipynb).
