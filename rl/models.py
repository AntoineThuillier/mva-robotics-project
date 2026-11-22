"""Policy pi and value function V (replaces MimicKit's PPOModel, BaseModel and
distribution_gaussian_diag.py). Article, section 6.2.

    pi(a | s, g) = N(mu(s, g), Sigma), Sigma = diag(sigma^2) fixed      -> GaussianPolicy
    V(s, g)                                                           -> ValueFunction

Both work on normalized observations, and pi on normalized actions (in [-1, 1] roughly, see
rl/ppo.py); s and g are concatenated by the environment into one observation vector.
"""

import math

import torch

from rl.nets import mlp


class Gaussian:
    """Diagonal Gaussian with mean (n, d) and log std (d,) or (n, d)."""

    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = torch.broadcast_to(logstd, mean.shape)
        self.std = torch.exp(self.logstd)

    @property
    def mode(self):
        return self.mean

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def log_prob(self, x):
        """log N(x; mean, std^2), summed over the d dimensions. (n,)."""
        d = self.mean.shape[-1]
        return (-0.5 * torch.sum(((x - self.mean) / self.std) ** 2, dim=-1)
                - 0.5 * d * math.log(2 * math.pi) - torch.sum(self.logstd, dim=-1))


class GaussianPolicy(torch.nn.Module):
    """MLP 1024-512 ReLU -> linear layer for the mean; fixed log std (not trained)."""

    def __init__(self, obs_size, action_size, action_std, init_output_scale=0.01):
        super().__init__()
        self.body = mlp(obs_size)
        self.mean_out = torch.nn.Linear(self.body[-2].out_features, action_size)
        # small initial means: the first actions stay close to the middle of the action range
        torch.nn.init.uniform_(self.mean_out.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self.mean_out.bias)
        self.register_buffer("logstd", torch.full((action_size,), math.log(action_std)))

    def forward(self, norm_obs):
        return Gaussian(self.mean_out(self.body(norm_obs)), self.logstd)


class ValueFunction(torch.nn.Module):
    """MLP 1024-512 ReLU -> one number."""

    def __init__(self, obs_size):
        super().__init__()
        self.body = mlp(obs_size)
        self.out = torch.nn.Linear(self.body[-2].out_features, 1)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, norm_obs):
        return self.out(self.body(norm_obs)).squeeze(-1)
