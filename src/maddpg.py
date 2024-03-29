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
    def __init__(self, config, index):
        self.tau = config.get('tau')
        seed = config.get('seed')
        # Change the seed a little according to the index to make agents got
        # different initial status.
        seed = (seed ** (index + 1)) % 1000000
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
        
    def _est(self, critic, state, action):
        if isinstance(state, np.ndarray):
            state = to_tensor(state)
        if isinstance(action, np.ndarray):
            action = to_tensor(action)
        critic.eval()
        with torch.no_grad():
            q = critic(torch.cat((state, action), 1)).cpu().data.numpy()
        critic.train()
        return q
        
    def est_local(self, state, action):
        return self._est(self.q_local, state, action)
        
    def est_target(self, state, action):
        return self._est(self.q_target, state, action)
        
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
        
        self._agents = [SingleDDPGAgent(config, 0) for i in range(self.agent_num)]
        self._agents[0].summary()
        
    def act(self, states):
        actions = np.zeros((self.agent_num, self.action_size))
        for i in range(self.agent_num):
            actions[i, :] = self._agents[i].act_local(states[i, :])
        return actions
        
    def calc_td_error(self, state, action, reward, next_state):
        local_states = get_local_view([state])
        local_next_states = get_local_view([next_state])
        local_actions = get_local_view([action])
        
        global_states = get_global_view([state])
        global_next_states = get_global_view([next_state])
        global_actions = get_global_view([action])

        td_err = np.zeros((self.agent_num, 1))
        mu_prime = [self._agents[i].act_target(local_next_states[i]) for i in range(self.agent_num)]
        global_mu = get_global_view([mu_prime])
        for i in range(self.agent_num):
            q_prime = self._agents[i].est_target(global_next_states, global_mu)
            q = self._agents[i].est_local(global_states, global_actions)
            td_err[i, 0] = reward[i] + q_prime * self.gamma - q
        return td_err

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
        mu_prime = [self._agents[i].mu_target(local_next_states[i]) for i in range(self.agent_num)]
        q_prime = agent.q_target(torch.cat([global_next_states] + mu_prime, 1).to(device))

        y = local_rewards[agent_index] + self.gamma * q_prime.squeeze() * (1. - local_dones[agent_index])
        y = y.detach()
        
        q = agent.q_local(torch.cat((global_states, global_actions), 1).to(device)).squeeze()
        td_err = y - q
        critic_loss = F.mse_loss(y, q)
        agent.q_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.q_local.parameters(), self.grad_clip)
        agent.q_optimizer.step()

        # Update the actor.
        mu = []
        for i in range(self.agent_num):
            mu_i = self._agents[i].mu_local(local_states[i])
            if i != agent_index:
                mu_i = mu_i.detach()
            mu.append(mu_i)

        policy_loss = -agent.q_local(torch.cat([global_states] + mu, 1).to(device)).mean()
        agent.mu_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(agent.mu_local.parameters(), self.grad_clip)
        agent.mu_optimizer.step()

        # Soft update the target networks.
        agent.soft_update()
        
        return (td_err.cpu().detach().numpy(),
                policy_loss.cpu().detach().item(),
                critic_loss.cpu().detach().item())


    def save_model(self, model_dir):
        for i in range(self.agent_num):
            self._agents[i].save(model_dir, i)
        