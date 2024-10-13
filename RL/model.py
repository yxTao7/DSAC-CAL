import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

EPS = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Initialize weights for ensemble networks
def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == torch.nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class Actor(torch.nn.Module):

    def __init__(self, state_size, action_size, hidden_size, seed=1):
        """Initialize parameters and build model. """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, action_size*2)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.gelu(self.fc1(state))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        y = self.fc_out(x)
        action_mean, action_log_std = torch.chunk(y, chunks=2, dim=-1)
        action_std = torch.clamp(action_log_std, -20, 0.5).exp()
        dist = Normal(action_mean, action_std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)
        log_prob = dist.log_prob(normal_sample) - torch.log(1 - torch.pow(action, 2) + EPS)
        return action, log_prob, action_mean


class Critic(torch.nn.Module):

    def __init__(self, state_size, action_size, hidden_size, seed):

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, 2)
        self.apply(weights_init_)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> action-value distribution."""

        cat = torch.cat([state, action], dim=-1)
        x = F.gelu(self.fc1(cat))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        y = self.fc_out(x)
        value_mean, value_std = torch.chunk(y, chunks=2, dim=-1)
        value_log_std = F.softplus(value_std)

        return value_mean, value_log_std


class EnsembleFC(torch.nn.Module):

    def __init__(self, in_features, out_features, ensemble_size, weight_decay):
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = torch.nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        self.bias = torch.nn.Parameter(torch.Tensor(ensemble_size, out_features))

    def forward(self, x):
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])


class QcNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, ensemble_size, hidden_size, seed):
        super(QcNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # 由三个集成全连接层组成
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.00003)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00006)
        self.nn3 = EnsembleFC(hidden_size, 1, ensemble_size, weight_decay=0.0001)
        self.activation = torch.nn.SiLU()
        self.ensemble_size = ensemble_size
        self.apply(init_weights)

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)
        # 输出扩展为ensemble_size个相同的副本
        nn1_output = self.activation(self.nn1(xu[None, :, :].repeat([self.ensemble_size, 1, 1])))
        nn2_output = self.activation(self.nn2(nn1_output))
        nn3_output = self.nn3(nn2_output)

        return nn3_output

