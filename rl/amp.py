"""AMP on our PPO (article sections 5 and 6.3, Algorithm 1; Table 4 for the settings).

    python rl/train.py --agent_config configs/amp_heading_humanoid_agent.yaml --out_dir outputs/rl/amp_heading

The discriminator D judges transitions (s, s') through the pair (Φ(s), Φ(s')) of
amp/disc_features.py (210 numbers): info["disc_obs"] for the agent, env.fetch_disc_obs_demo(n)
for the mocap (amp/heading_env.py, amp/expert.py). Its score gives the style reward r_S, added
to the task reward before GAE: r = w_G r_G + w_S r_S (eq. 4).

AMPTrainer = Trainer (rl/train.py) + three hooks, one iteration:
  record_step     every policy step: store the agent's disc_obs with the transition;
  before_targets  after the rollout: r_S = style_reward(D(Φ)) for every transition, the reward
                  becomes w_G r_G + w_S r_S; disc_replay_insert random agent pairs go into the
                  replay buffer B;
  after_update    after PPO: disc_steps updates of D, each on disc_batch_size fresh mocap pairs
                  and as many agent pairs drawn from B, loss = eq. 6 + eq. 8.

Batch sizes adapted to 4096 GPU envs (131k transitions per iteration instead of the article's
4096): B keeps the article's size (1e5) and receives 4096 pairs per iteration, so it spans ~25
iterations as in the article; D sees as many agent pairs per iteration as the rollout produced.
D's input is normalized with a mean and std computed once on disc_norm_samples mocap pairs, then
frozen. Same algorithm as amp/discriminator.py (built on MimicKit's PPO), checked against it by
validation/rl_check.py, test 5.
"""

import torch

from rl.nets import mlp
from rl.normalizer import Normalizer
from rl.ppo import build_optimizer
from rl.train import Trainer


# ===================== the article's equations =====================

def disc_loss_lsgan(d_mocap, d_agent):
    """Eq. 6, least-squares GAN: D is regressed to +1 on mocap transitions and -1 on the agent's."""
    return torch.mean((d_mocap - 1) ** 2) + torch.mean((d_agent + 1) ** 2)


def gradient_penalty(disc_fn, phi_mocap, w_gp):
    """Eq. 8, last term: (w_gp / 2) E_mocap[ ||grad_phi D(phi)||^2 ], on mocap transitions only,
    with respect to D's input (create_graph=True: the penalty is then minimized over D's weights).
    disc_fn: phi (n, 210) -> D (n,)."""
    phi = phi_mocap.detach().requires_grad_(True)
    grad = torch.autograd.grad(disc_fn(phi).sum(), phi, create_graph=True)[0]
    return 0.5 * w_gp * torch.mean(torch.sum(grad ** 2, dim=-1))


def style_reward(d):
    """Eq. 7: r_S = max(0, 1 - 0.25 (D - 1)^2), in [0, 1]."""
    return torch.clamp_min(1 - 0.25 * (d - 1) ** 2, 0.0)


# ===================== discriminator and replay buffer =====================

class Discriminator(torch.nn.Module):
    """MLP 1024-512 ReLU (as pi and V, article section 6.2), then a linear output, no sigmoid (LSGAN)."""

    def __init__(self, in_size):
        super().__init__()
        self.body = mlp(in_size)
        self.out = torch.nn.Linear(self.body[-2].out_features, 1)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, norm_disc_obs):
        return self.out(self.body(norm_disc_obs)).squeeze(-1)


class ReplayBuffer:
    """FIFO of agent disc observations (the article's B): the oldest pairs are overwritten."""

    def __init__(self, size, dim, device):
        self.data = torch.zeros(size, dim, device=device)
        self.head = 0
        self.count = 0

    def push(self, x):
        size = self.data.shape[0]
        idx = (self.head + torch.arange(x.shape[0], device=x.device)) % size
        self.data[idx] = x
        self.head = (self.head + x.shape[0]) % size
        self.count = min(self.count + x.shape[0], size)

    def sample(self, n):
        return self.data[torch.randint(0, self.count, (n,), device=self.data.device)]

    def __len__(self):
        return self.count


# ===================== trainer =====================

class AMPTrainer(Trainer):

    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        assert config["model"].get("disc_net", "fc_2layers_1024units") == "fc_2layers_1024units"
        self.task_reward_weight = float(config["task_reward_weight"])     # w_G
        self.style_reward_weight = float(config["style_reward_weight"])   # w_S
        self.w_gp = float(config["disc_grad_penalty"])
        self.disc_batch_size = int(config["disc_batch_size"])
        self.disc_steps = int(config["disc_steps"])
        self.disc_replay_insert = int(config["disc_replay_insert"])

        dim = env.get_disc_obs_space().shape[0]
        self.disc = Discriminator(dim).to(device)
        self.disc_optimizer = build_optimizer(config["disc_optimizer"], self.disc.parameters())
        self.replay = ReplayBuffer(int(config["disc_replay_size"]), dim, device)

        # D's input normalizer: mocap pairs only, once, then frozen
        self.disc_norm = Normalizer(dim, device, clip=10.0)
        n, total = 0, int(config["disc_norm_samples"])
        while n < total:
            batch = min(65536, total - n)
            self.disc_norm.record(env.fetch_disc_obs_demo(batch))
            n += batch
        self.disc_norm.update()

    @torch.no_grad()
    def eval_disc(self, disc_obs, batch_size=65536):
        """D on raw disc observations, by minibatches, without gradient. (n,)."""
        return torch.cat([self.disc(self.disc_norm.normalize(disc_obs[i:i + batch_size]))
                          for i in range(0, disc_obs.shape[0], batch_size)])

    # ---------- hooks ----------
    def record_step(self, next_info):
        self.buf.record("disc_obs", next_info["disc_obs"])

    def before_targets(self):
        disc_obs = self.buf.flat("disc_obs")                                # (T * N, 210)
        task_r = self.buf.data["reward"]
        style_r = style_reward(self.eval_disc(disc_obs)).view_as(task_r)
        self.buf.set("reward", self.task_reward_weight * task_r + self.style_reward_weight * style_r)
        idx = torch.randperm(disc_obs.shape[0], device=self.device)[:self.disc_replay_insert]
        self.replay.push(disc_obs[idx])
        return {"task_reward_mean": task_r.mean(), "style_reward_mean": style_r.mean()}

    def after_update(self):
        """disc_steps updates of D on eq. 6 + eq. 8. Returns the means of the diagnostics."""
        self.disc.train()
        sums = {}
        for _ in range(self.disc_steps):
            mocap = self.disc_norm.normalize(self.env.fetch_disc_obs_demo(self.disc_batch_size))
            agent = self.disc_norm.normalize(self.replay.sample(self.disc_batch_size))
            step = self.disc_loss(mocap, agent)
            self.disc_optimizer.zero_grad()
            (step["disc_loss"] + step["disc_grad_penalty"]).backward()
            self.disc_optimizer.step()
            for k, v in step.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
        self.disc.eval()
        return {k: v / self.disc_steps for k, v in sums.items()}

    def disc_loss(self, norm_mocap, norm_agent):
        """Losses and diagnostics of D on normalized mocap and agent pairs."""
        d_mocap, d_agent = self.disc(norm_mocap), self.disc(norm_agent)
        return {"disc_loss": disc_loss_lsgan(d_mocap, d_agent),
                "disc_grad_penalty": gradient_penalty(self.disc, norm_mocap, self.w_gp),
                "disc_acc_mocap": (d_mocap > 0).float().mean(),             # mocap judged mocap
                "disc_acc_agent": (d_agent < 0).float().mean(),             # agent judged agent
                "disc_mocap_mean": d_mocap.mean(), "disc_agent_mean": d_agent.mean()}

    # ---------- saving ----------
    def extra_state(self):
        return {"disc": self.disc.state_dict(), "disc_norm": self.disc_norm.state_dict()}

    def load(self, path):
        state = super().load(path)
        if "disc" in state:
            self.disc.load_state_dict(state["disc"])
            self.disc_norm.load_state_dict(state["disc_norm"])
        return state
