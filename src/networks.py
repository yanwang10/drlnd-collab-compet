import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
def layer_init(layer, w_scale):
    """
    Copied from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_utils.py
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class BasicNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[], nonlinear=None, seed=13173, init_weight_scale=1.0):
        super(BasicNetwork, self).__init__()
        dims = [in_dim] + hidden + [out_dim]
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dims[i], dims[i + 1]), init_weight_scale) for i in range(len(dims) - 1)])
        self.nonlinear = nonlinear
        self.seed = torch.manual_seed(seed)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.nonlinear:
                x = self.nonlinear(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[], out_range=(None,None), seed=13173, init_weight_scale=1.0):
        super(ActorNetwork, self).__init__()
        assert(len(hidden) > 0)
        self.feature = BasicNetwork(in_dim, hidden[-1], hidden[:-1], F.relu, seed, init_weight_scale)
        self.output = BasicNetwork(hidden[-1], out_dim, nonlinear=F.tanh, seed=seed, init_weight_scale=init_weight_scale)
        self.out_low, self.out_high = out_range
        assert((self.out_low is None) == (self.out_high is None))
        
    def forward(self, x):
        # print('forwarding actor feature')
        x = self.feature(x)
        # print('forwarding actor output')
        x = self.output(x)
        if not self.out_low is None:
            x = torch.clamp(x, min=self.out_low, max=self.out_high)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=[], seed=13173, init_weight_scale=1.0):
        super(CriticNetwork, self).__init__()
        assert(len(hidden) > 0)
        self.feature = BasicNetwork(in_dim, hidden[-1], hidden[:-1], F.relu, seed, init_weight_scale)
        self.output = BasicNetwork(hidden[-1], out_dim, seed=seed, init_weight_scale=init_weight_scale)
        
    def forward(self, x):
        # print('forwarding critic feature')
        x = self.feature(x)
        # print('forwarding critic output')
        return self.output(x)
