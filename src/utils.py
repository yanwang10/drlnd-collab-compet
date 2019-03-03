import numpy as np
from collections import deque, namedtuple
import copy, time, random, json, os, heapq
import torch
import torch.nn as nn
import pickle
from operator import itemgetter
from tensorboardX import SummaryWriter
from .per import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_tensor(np_array):
    """
    Converts an numpy array to torch tensor.
    """
    return torch.from_numpy(np_array).float().to(device)

def get_local_view(arrs, as_tensor=True):
    """
    Given a list of B numpy arrays of size (N, X), returns a list of N
    numpy arrays of size (B, X).
    """
    # Stack all arrays to a big array with size (N, B, X)
    stacked = np.stack([np.expand_dims(arr, 1) for arr in arrs], 1)
    # Split the first dimension.
    views = [np.squeeze(stacked[i, :, :]) for i in range(stacked.shape[0])]
    if as_tensor:
        for i in range(len(views)):
            views[i] = to_tensor(views[i])
    return views

def get_global_view(arrs, as_tensor=True):
    """
    Given a list of B numpy arrays of size (N, X), returns a new array of
    size (B, N*X).
    """
    view = np.vstack([np.reshape(arr, (1, -1)) for arr in arrs])
    if as_tensor:
        view = to_tensor(view)
    return view
        

class ConfigurableObject:
    """
    A base class of all objects that could be configured with a flattened dict.
    The values could be objects or functions/lambdas.
    """
    def __init__(self, config, default={}):
        self.config = copy.deepcopy(config)
        self.default = copy.deepcopy(default)

    def has(self, k):
        return k in self.config or k in self.default

    def get(self, k):
        if k in self.config:
            return self.config[k]
        elif k in self.default:
            return self.default[k]
        else:
            return None

    def print_config(self):
        config = self.default
        for k in self.config:
            config[k] = self.config[k]
        print('Configs: ', json.dumps(config, indent=4))


class SmoothAccumulator:
    """
    An accumulator that records the raw values and smoothed values within a
    certain window.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self._raw = list()
        self._smooth = list()
        self._window = deque(maxlen=self.window_size)

    def add(self, v):
        self._raw.append(v)
        self._window.append(v)
        self._smooth.append(np.mean(self._window))

    def get_latest_record(self):
        return self._raw[-1], self._smooth[-1]

    def get_all_records(self):
        return self._raw, self._smooth


class RLTrainingLogger:
    """
    A helper class that logs the rewards and running time of training process.
    """
    def __init__(self, window_size=100, log_file=None, log_interval=50):
        self.reward_log = SmoothAccumulator(window_size)
        self.time_log = SmoothAccumulator(window_size)
        self.step_log = SmoothAccumulator(window_size)
        self._episode_count = 0
        self._step_count = 0
        self._start_timestamp = None
        self._log_file = log_file
        self._log_interval = log_interval

    def episode_begin(self):
        self._start_timestamp = time.time()

    def episode_end(self, reward, step_num):
        time_interval = time.time() - self._start_timestamp
        self._start_timestamp = None
        self.reward_log.add(reward)
        self.time_log.add(time_interval)
        self.step_log.add(step_num)
        self._step_count += step_num
        self._episode_count += 1
        _, smooth_r = self.reward_log.get_latest_record()
        _, smooth_t = self.time_log.get_latest_record()
        _, smooth_s = self.step_log.get_latest_record()
        print('\repisode: %6d, step: %7d, reward: %.3f, avg_step: %3d, avg_time: %.2fs' %
              (self._episode_count, self._step_count, smooth_r, smooth_s, smooth_t), end='')
        if self._episode_count % self._log_interval == 0:
            print('')
            if self._log_file:
                with open(self._log_file, 'wb') as f:
                    pickle.dump(self, f)

    def get_all_rewards(self):
        return self.reward_log.get_all_records()


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Copied from the Udacity DQN mini project.
    """

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = Memory(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, err):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, np.array(done).astype(np.uint8))
        self.memory.add(err, e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences, idxs, _ = self.memory.sample(self.batch_size)
        
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        next_states = [e.next_state for e in experiences]
        rewards = [e.reward for e in experiences]
        dones = [e.done for e in experiences]
        
        return idxs, (states, actions, next_states, rewards, dones)

    def merge_from(self, another_buffer):
        for e in another_buffer.memory:
            self.memory.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory.tree.n_entries


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    Based on the implementation in Udacity DDPG mini project:
    https://github.com/udacity/deep-reinforcement-learning/blob/d6cb43c1b11b1d55c13ac86d6002137c7b880c15/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, discount=1.0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.discount = discount
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.coef = 1.

    def add_to(self, v):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        self.coef *= self.discount
        return self.state * self.coef * random.choice([1., -1.]) + v * (1. - self.coef)


def TrainMADDPG(env, agent, config):
    
    # Copy the configs.
    agent_num = config['agent_num']
    action_repeat = config['action_repeat']
    action_size = config['action_size']
    state_size = config['state_size'] * action_repeat
    seed = config['seed']
    batch_num = config['batch_num']
    max_step_num = config['max_step_num']
    max_episode_num = config['max_episode_num']
    learn_interval = config['learn_interval']
    out_low = config['out_low']
    out_high = config['out_high']
    lambda_return = config['lambda_return']
    warmup_step = config['warmup_step']
    
    # Prepare the utilities.
    os.makedirs(config['model_dir'], exist_ok=True)
    buffer = ReplayBuffer(config['buffer_size'], batch_num, seed)
    noise = OUNoise((agent_num, action_size), seed, discount=config['noise_discount'])
    logger = RLTrainingLogger(config['window_size'], config['log_file'], config['log_interval'])
    tb_logger = SummaryWriter(log_dir=config['tensorboard_log_dir'])
    
    # The main training process.
    total_step_num = 0
    total_episode_num = 0
    
    get_last_repeated_state = lambda states: np.concatenate(states[-action_repeat:], 1)
    
    best_performance = None
    
    while total_step_num < max_step_num and total_episode_num < max_episode_num:
        # Start an episode
        raw_state = env.reset()
        episode_done = False
        logger.episode_begin()
        state = np.concatenate([raw_state for _ in range(action_repeat)], 1)
        episode_rewards = []
        episode_buffer = []
        beginning_step = total_step_num
        while not episode_done:
            action = noise.add_to(agent.act(state))
            action = np.clip(action, a_min=out_low, a_max=out_high)
            states = []
            for _ in range(action_repeat):
                next_raw_state, reward, done = env.step(action)
                total_step_num += 1
                states.append(next_raw_state)
                episode_rewards.append(reward)
                if any(done):
                    break
            # print('Interacting: next_state.shape =', next_state.shape)
            episode_done = any(done)
            if len(states) == action_repeat:
                next_state = np.concatenate(states, 1)
                acc_reward = sum(episode_rewards[-action_repeat:])
                td_err = agent.calc_td_error(state, action, acc_reward, next_state)
                episode_buffer.append((state, action, acc_reward, next_state, done, sum(np.absolute(td_err))))
            state = next_state
        
            # Update params from experience replay.
            if len(buffer) > batch_num and total_step_num > warmup_step:# and total_episode_num % learn_interval == 0:
                idxs, experiences = buffer.sample()
                td_err = np.zeros((batch_num, agent_num))
                for i in range(agent_num):
                    err, policy_loss, critic_loss = agent.learn(i, experiences)
                    td_err[:, i] = err
                    tb_logger.add_scalars('loss/critic/',
                           {'agent%d' % i: critic_loss}, total_step_num)
                    tb_logger.add_scalars('loss/policy/', {
                            'agent%d' % i: policy_loss}, total_step_num)
                for i, idx in enumerate(idxs):
                    buffer.memory.update(idx, sum(np.absolute(td_err[i, :])))
                    tb_logger.add_scalars('td_err/',
                        {'step': sum(np.absolute(td_err[i, :]))},
                         batch_num * total_episode_num + i)
            
        # Calculate the lambda-return and store the results into global replay buffer.
        discounted_return = None
        for s, a, r, ns, d, err in reversed(episode_buffer):
            if discounted_return is None:
                discounted_return = r
            else:
                discounted_return = r + discounted_return * lambda_return
            buffer.add(s, a, discounted_return, ns, d, err)


        # Take the max reward over all agents as the final reward of the episode.
        episode_reward = max(sum(episode_rewards))
        logger.episode_end(episode_reward, total_step_num - beginning_step)
        total_episode_num += 1
        
        # Save the model as long as its recent smoothed reward is higher than
        # the previous best performance by some margin.
        smooth_performance = logger.reward_log.get_latest_record()[1]
        if best_performance is None or smooth_performance > best_performance + .01:
            agent.save_model(config.get('model_dir'))
            best_performance = smooth_performance

        tb_logger.add_scalars(
            'reward/episode',
            {'smooth_reward': smooth_performance},
            total_episode_num)