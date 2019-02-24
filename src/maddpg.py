import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import *
from .utils import to_tensor, get_local_view, get_global_view
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def soft_update(src, dst, tau):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.detach_()
        dst_param.data.copy_(
            tau * src_param.data + (1.0 - tau) * dst_param.data)

class SingleDDPGAgent(object):
    def __init__(self, config):
        self.tau = config.get('tau')
        seed = config.get('seed')
        agent_num = config.get('agent_num')
        actor_hidden = config.get('actor_hidden')
        critic_hidden = config.get('critic_hidden')
        init_weight_scale = config.get('init_weight_scale')

        action_range = [config.get('out_low'), config.get('out_high')]
        action_size = config.get('action_size')
        
        action_repeat = config.get('action_repeat')
        state_size = config.get('state_size')
        # Each actor can only see its own observation.
        self.actor_input_size = state_size * action_repeat
        # The critics can see all observations and actions.
        self.critic_input_size = (self.actor_input_size + action_size) * agent_num
        
        # For some unknown reason copy.deepcopy fails to copy a model with error
        # message: TypeError: can't pickle torch._C.Generator objects
        # So we have to create new networks with the same structure and copy the
        # weights.
        self.q_local = CriticNetwork(self.critic_input_size, 1, critic_hidden, seed=seed,
            init_weight_scale=init_weight_scale).to(device)
        self.q_target = CriticNetwork(self.critic_input_size, 1, critic_hidden, seed=seed,
            init_weight_scale=init_weight_scale).to(device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.q_optimizer = optim.Adam(self.q_local.parameters(), lr=config.get('critic_lr'))
        
        self.mu_local = ActorNetwork(self.actor_input_size, action_size, actor_hidden,
            action_range, seed, init_weight_scale=init_weight_scale).to(device)
        self.mu_target = ActorNetwork(self.actor_input_size, action_size, actor_hidden,
            action_range, seed, init_weight_scale=init_weight_scale).to(device)
        self.mu_target.load_state_dict(self.mu_local.state_dict())
        self.mu_optimizer = optim.Adam(self.mu_local.parameters(), lr=config.get('actor_lr'))

    def summary(self):
        summary(self.q_local, (self.critic_input_size,))
        summary(self.mu_local, (self.actor_input_size,))
        
        
    def _act(self, actor, state):
        if isinstance(state, np.ndarray):
            state = to_tensor(state)
        actor.eval()
        with torch.no_grad():
            action = actor(state).cpu().data.numpy()
        actor.train()
        return action

    def act_local(self, state):
        return self._act(self.mu_local, state)
    
    def act_target(self, state):
        return self._act(self.mu_target, state)
        
    def save(self, model_dir, index):
        torch.save(self.q_local.state_dict(), model_dir + '/q_local_%d.pt' % index)
        torch.save(self.q_target.state_dict(), model_dir + '/q_target_%d.pt' % index)
        torch.save(self.mu_local.state_dict(), model_dir + '/mu_local_%d.pt' % index)
        torch.save(self.mu_target.state_dict(), model_dir + '/mu_target_%d.pt' % index)

    def soft_update(self):
        soft_update(self.q_local, self.q_target, self.tau)
        soft_update(self.mu_local, self.mu_target, self.tau)


class MADDPGAgents(object):
    """
    Implementation of DDPG algorithm described in this paper:
    Continuous Control with Deep Reinforcement Learning
    https://arxiv.org/pdf/1509.02971.pdf
    """
    def __init__(self, config):
        self.gamma = config.get('gamma')
        self.agent_num = config.get('agent_num')
        self.grad_clip = config.get('grad_clip')
        self.action_size = config.get('action_size')
        
        self._agents = [SingleDDPGAgent(config) for _ in range(self.agent_num)]
        self._agents[0].summary()
        
    def act(self, states):
        actions = np.zeros((self.agent_num, self.action_size))
        for i in range(self.agent_num):
            actions[i, :] = self._agents[i].act_local(states[i, :])
        return actions

    def learn(self, agent_index, experiences):
        agent = self._agents[agent_index]
        raw_states, raw_actions, raw_next_states, raw_rewards, raw_dones = experiences
        
        # Reformat the experiences
        local_states = get_local_view(raw_states)
        local_next_states = get_local_view(raw_next_states)
        local_actions = get_local_view(raw_actions)
        local_rewards = get_local_view(raw_rewards)
        local_dones = get_local_view(raw_dones)
        
        global_states = get_global_view(raw_states)
        global_next_states = get_global_view(raw_next_states)
        global_actions = get_global_view(raw_actions)
        
        # Update the critic.
        mu_prime = [self._agents[i].mu_target(local_states[i]) for i in range(self.agent_num)]
        q_prime = agent.q_target(torch.cat([global_next_states] + mu_prime, 1))

        y = local_rewards[agent_index] + self.gamma * q_prime * (1. - local_dones[agent_index])
        y = y.detach()
        
        q = agent.q_local(torch.cat((global_states, global_actions), 1))
        critic_loss = F.mse_loss(y, q)
        agent.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.q_local.parameters(), self.grad_clip)
        agent.q_optimizer.step()

        # Update the actor.
        predicted_actions = agent.mu_local(local_states[agent_index])
        mu_prime[agent_index] = predicted_actions
        policy_loss = -agent.q_local(torch.cat([global_states] + mu_prime, 1)).mean()
        agent.mu_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(agent.mu_local.parameters(), self.grad_clip)
        agent.mu_optimizer.step()

        # Soft update the target networks.
        agent.soft_update()


    def save_model(self, model_dir):
        for i in range(self.agent_num):
            self._agents[i].save(model_dir, i)
        