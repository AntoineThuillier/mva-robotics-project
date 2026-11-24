"""PPO with continuous actions (replaces MimicKit's learning/ppo_agent.py). Schulman et al. 2017.

    ppo = PPO(config, obs_size, action_low, action_high, device)
    a, logp = ppo.act(obs, explore=True)          # one action per environment
    ppo.compute_targets(buf, reward_fail, reward_succ)
    info = ppo.update(buf)                         # critic, then actor

Reproduces MimicKit's PPO exactly, with its details that the article does not give:
  - actions are handled in a normalized space: a_norm = (a - mid) / half_range of the action
    bounds of the environment; pi is a Gaussian there; a penalty (action_bound_weight) keeps its
    mean inside [-1, 1];
  - observations are normalized by a running mean/std (rl/normalizer.py), updated by the
    training loop during the first normalizer_samples samples, then frozen;
  - advantages: TD(lambda) return - V(s), normalized over the iteration, clipped to +-norm_adv_clip;
  - minibatch sizes are given per environment: batch = batch_size x num_envs samples; the
    number of steps covers the buffer `epochs` times;
  - SGD with momentum 0.9.
MimicKit's options not used by our configs (exploration probability < 1, entropy and action
regularization, mixed precision, several GPUs) are left out; the constructor checks they are off.
"""

import numpy as np
import torch

from rl.models import GaussianPolicy, ValueFunction
from rl.normalizer import Normalizer
from rl.returns import end_values, td_lambda_returns


def build_optimizer(config, params):
    """MimicKit's optimizers: SGD (momentum 0.9) or AdamW, with an optional weight_decay
    (default 0; MimicKit's AMP uses 1e-4 on D, the article none)."""
    lr = float(config["learning_rate"])
    wd = float(config.get("weight_decay", 0.0))
    if config["type"] == "SGD":
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=wd)
    if config["type"] == "Adam":
        return torch.optim.AdamW(params, lr, weight_decay=wd)
    raise ValueError(f"unsupported optimizer {config['type']}")


class PPO(torch.nn.Module):

    def __init__(self, config, obs_size, action_low, action_high, num_envs, device):
        super().__init__()
        assert config.get("exp_prob_beg", 1.0) == 1.0 and config.get("exp_prob_end", 1.0) == 1.0
        assert config.get("action_entropy_weight", 0.0) == 0.0 and config.get("action_reg_weight", 0.0) == 0.0
        assert not config.get("use_mixed_precision", False)
        model = config["model"]
        assert model["actor_net"] == model["critic_net"] == "fc_2layers_1024units"
        assert model["actor_std_type"] == "FIXED"

        self.device = device
        self.num_envs = num_envs
        self.discount = float(config["discount"])
        self.td_lambda = float(config["td_lambda"])
        self.clip_ratio = float(config["ppo_clip_ratio"])
        self.norm_adv_clip = float(config["norm_adv_clip"])
        self.action_bound_weight = float(config["action_bound_weight"])
        self.actor_epochs, self.actor_batch_size = int(config["actor_epochs"]), float(config["actor_batch_size"])
        self.critic_epochs, self.critic_batch_size = int(config["critic_epochs"]), float(config["critic_batch_size"])

        low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.obs_norm = Normalizer(obs_size, device, clip=10.0)
        self.action_norm = Normalizer(low.shape[0], device, clip=np.inf,
                                      init_mean=0.5 * (high + low), init_std=0.5 * (high - low))
        self.policy = GaussianPolicy(obs_size, low.shape[0], float(model["action_std"]),
                                     float(model["actor_init_output_scale"])).to(device)
        self.value = ValueFunction(obs_size).to(device)
        self.actor_optimizer = build_optimizer(config["actor_optimizer"], self.policy.parameters())
        self.critic_optimizer = build_optimizer(config["critic_optimizer"], self.value.parameters())

    # ---------- acting ----------
    @torch.no_grad()
    def act(self, obs, explore):
        """explore: sample pi (training) or take its mean (test). Returns the action in the
        environment's units and log pi(a_norm | s) of the normalized action."""
        dist = self.policy(self.obs_norm.normalize(obs))
        a_norm = dist.sample() if explore else dist.mode
        return self.action_norm.unnormalize(a_norm), dist.log_prob(a_norm)

    # ---------- targets: TD(lambda) returns and advantages ----------
    @torch.no_grad()
    def compute_targets(self, buf, reward_fail, reward_succ):
        """Adds "tar_val" (TD(lambda) return) and "adv" (normalized, clipped advantage) to the buffer."""
        r, done = buf.data["reward"], buf.data["done"]
        next_vals = self.value(self.obs_norm.normalize(buf.data["next_obs"]))
        next_vals = end_values(next_vals, done, self.discount, reward_fail, reward_succ)
        ret = td_lambda_returns(r, next_vals, done, self.discount, self.td_lambda)
        adv = ret - self.value(self.obs_norm.normalize(buf.data["obs"]))
        adv_std, adv_mean = torch.std_mean(adv.flatten())
        norm_adv = torch.clamp((adv - adv_mean) / torch.clamp_min(adv_std, 1e-5), -self.norm_adv_clip, self.norm_adv_clip)
        buf.set("tar_val", ret)
        buf.set("adv", norm_adv)
        return {"adv_mean": adv_mean, "adv_std": adv_std}

    # ---------- losses ----------
    def critic_loss(self, batch):
        pred = self.value(self.obs_norm.normalize(batch["obs"]))
        return {"critic_loss": torch.mean((batch["tar_val"] - pred) ** 2)}

    def actor_loss(self, batch):
        dist = self.policy(self.obs_norm.normalize(batch["obs"]))
        a_logp = dist.log_prob(self.action_norm.normalize(batch["action"]))
        ratio = torch.exp(a_logp - batch["a_logp"])
        adv = batch["adv"]
        loss = -torch.mean(torch.minimum(adv * ratio, adv * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)))
        # keep the mean action inside the normalized bounds [-1, 1]
        bound = torch.mean(torch.sum(torch.clamp_max(dist.mode + 1, 0) ** 2, -1)
                           + torch.sum(torch.clamp_min(dist.mode - 1, 0) ** 2, -1))
        return {"actor_loss": loss + self.action_bound_weight * bound,
                "clip_frac": torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float()).detach(),
                "imp_ratio": torch.mean(ratio).detach(),
                "action_bound_loss": bound.detach()}

    # ---------- update ----------
    def update(self, buf):
        """Critic first, then actor, as MimicKit. Returns the mean of the losses over the steps."""
        self.train()
        n = buf.steps * buf.num_envs
        info = {}
        for loss_fn, opt, epochs, batch, key in [
                (self.critic_loss, self.critic_optimizer, self.critic_epochs, self.critic_batch_size, "critic_loss"),
                (self.actor_loss, self.actor_optimizer, self.actor_epochs, self.actor_batch_size, "actor_loss")]:
            batch_size = int(np.ceil(batch * self.num_envs))
            steps = int(np.ceil(n / batch_size)) * epochs
            sums = {}
            for mb in buf.minibatches(batch_size, steps):
                out = loss_fn(mb)
                opt.zero_grad()
                out[key].backward()
                opt.step()
                for k, v in out.items():
                    sums[k] = sums.get(k, 0.0) + v.detach()
            info.update({k: v / steps for k, v in sums.items()})
        self.eval()
        return info
