"""Our AMP agent (phase 4, CADRAGE.md; article sections 5 and 6.3, Algorithm 1).

    python amp/run.py --arg_file configs/amp_heading_humanoid_args.txt

Built on MimicKit's PPO (option A of phase 4): PPO itself (rollouts, GAE, actor and critic
updates, logs, saving) is MimicKit's `PPOAgent`; what the article adds is written here.

    HeadingAMPModel   MimicKit's PPOModel (actor pi, critic V)      + discriminator D  [step 3]
    HeadingAMPAgent   MimicKit's PPOAgent                           + style reward r_S, replay
                                                                      buffer, update of D [step 4]
    disc_loss_lsgan, gradient_penalty, style_reward: the article's equations 6, 8 and 7 [step 3]

One training iteration (Algorithm 1), with MimicKit's loop:
  1. rollout, 32 policy steps x num_envs: the env's info["disc_obs"] (agent pairs) is stored
     with each transition (_record_data_post_step);
  2. before GAE (_build_train_data): r_S = style_reward(D(Φ)) for every transition, then
     r = w_G r_G + w_S r_S replaces the reward; disc_replay_insert random agent pairs go into
     the replay buffer B;
  3. PPO update of pi and V (MimicKit, unchanged), then disc_steps updates of D (_update_disc):
     minibatches of disc_batch_size mocap pairs (fresh from amp/expert.py) and as many agent
     pairs drawn from B, loss = eq. 6 + eq. 8.

Batch sizes, adapted to 4096 GPU envs (131k transitions per iteration instead of the article's
4096): B keeps the article's size (1e5) and receives 4096 pairs per iteration, as in the article,
so it spans ~25 iterations; D sees as many agent pairs per iteration as the rollout produced.

D's input (Φ(s), Φ(s')) is normalized with a mean and std computed once, at construction, on
disc_norm_samples mocap pairs, then frozen (the mocap distribution does not change; the agent's
does). The gradient penalty is taken with respect to this normalized input.

The discriminator sees a transition as the pair (Φ(s), Φ(s')) of amp/disc_features.py (210
numbers), given by the env as info["disc_obs"] for the agent and by fetch_disc_obs_demo for the
mocap (amp/heading_env.py).

Selected with `agent_name: "AMP_heading"` in the agent config; amp/run.py only routes that
name to `build_heading_amp_agent` below. Like the rest of amp/, import with MimicKit's
`mimickit/` folder in sys.path.
"""

import torch

import learning.mp_optimizer as mp_optimizer
import learning.nets.net_builder as net_builder
import learning.normalizer as normalizer
import learning.ppo_agent as ppo_agent
import learning.ppo_model as ppo_model
import util.torch_util as torch_util

AGENT_NAME = "AMP_heading"


# ===================== the article's equations (tested in validation/phase4.py, test 2) =====================

def disc_loss_lsgan(d_mocap, d_agent):
    """Eq. 6, least-squares GAN: D is regressed to +1 on mocap transitions and -1 on the agent's.
    d_mocap, d_agent: outputs of D (no sigmoid), any shape. Returns a scalar."""
    return torch.mean((d_mocap - 1) ** 2) + torch.mean((d_agent + 1) ** 2)


def gradient_penalty(disc_fn, phi_mocap, w_gp):
    """Eq. 8, last term: (w_gp / 2) * E_mocap[ ||grad_phi D(phi)||^2 ], on mocap transitions only.

    The gradient is taken with respect to the input of D (the Φ pair, normalized as D sees it),
    not to D's weights; create_graph=True so that the penalty itself can be minimized over D's
    weights. disc_fn: phi (n, 210) -> D (n,). Returns a scalar."""
    phi = phi_mocap.detach().requires_grad_(True)
    d = disc_fn(phi)
    grad = torch.autograd.grad(d.sum(), phi, create_graph=True)[0]      # (n, 210): dD/dphi per sample
    return 0.5 * w_gp * torch.mean(torch.sum(grad ** 2, dim=-1))


def style_reward(d):
    """Eq. 7: r_S = max(0, 1 - 0.25 (D - 1)^2), in [0, 1]: 1 for D = +1 (mocap-like), 0 for D <= -1."""
    return torch.clamp_min(1 - 0.25 * (d - 1) ** 2, 0.0)


# ===================== model and agent =====================

class HeadingAMPModel(ppo_model.PPOModel):
    """Actor pi and critic V (MimicKit's PPOModel) + the discriminator D.

    D: MLP named by config["disc_net"] (fc_2layers_1024units: 1024 and 512 ReLU, as pi and V,
    article section 6.2), then a linear output of 1 number, without sigmoid (LSGAN)."""

    def __init__(self, config, env):
        super().__init__(config, env)
        return

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._build_disc(config, env)
        return

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        self._disc_layers, _ = net_builder.build_net(config["disc_net"], input_dict, activation=self._activation)
        self._disc_out = torch.nn.Linear(torch_util.calc_layers_out_size(self._disc_layers), 1)
        torch.nn.init.zeros_(self._disc_out.bias)
        return

    def eval_disc(self, norm_disc_obs):
        """D(Φ(s), Φ(s')) for normalized disc observations (n, 210). Returns (n,)."""
        return self._disc_out(self._disc_layers(norm_disc_obs)).squeeze(-1)

    def get_disc_params(self):
        return list(self._disc_layers.parameters()) + list(self._disc_out.parameters())


class ReplayBuffer:
    """FIFO of agent disc observations (the article's B): the oldest pairs are overwritten."""

    def __init__(self, size, dim, device):
        self._data = torch.zeros(size, dim, device=device)
        self._head = 0
        self._count = 0

    def push(self, x):
        n, size = x.shape[0], self._data.shape[0]
        idx = (self._head + torch.arange(n, device=x.device)) % size
        self._data[idx] = x
        self._head = (self._head + n) % size
        self._count = min(self._count + n, size)

    def sample(self, n):
        return self._data[torch.randint(0, self._count, (n,), device=self._data.device)]

    def __len__(self):
        return self._count


class HeadingAMPAgent(ppo_agent.PPOAgent):
    """MimicKit's PPOAgent + the style reward and the training of D (Algorithm 1, see the module docstring)."""

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    # ---------- construction ----------
    def _load_params(self, config):
        super()._load_params(config)
        self._task_reward_weight = float(config["task_reward_weight"])     # w_G
        self._style_reward_weight = float(config["style_reward_weight"])   # w_S
        self._disc_grad_penalty = float(config["disc_grad_penalty"])       # w_gp
        self._disc_batch_size = int(config["disc_batch_size"])             # pairs per source and per step
        self._disc_steps = int(config["disc_steps"])                       # updates of D per iteration
        self._disc_replay_size = int(config["disc_replay_size"])
        self._disc_replay_insert = int(config["disc_replay_insert"])       # agent pairs added to B per iteration
        self._disc_norm_samples = int(config["disc_norm_samples"])
        return

    def _build_model(self, config):
        self._model = HeadingAMPModel(config["model"], self._env)
        return

    def _build_optimizer(self, config):
        super()._build_optimizer(config)                                    # pi and V (D excluded)
        self._disc_optimizer = mp_optimizer.MPOptimizer(config["disc_optimizer"], self._model.get_disc_params())
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._disc_optimizer.sync()
        return

    def _build_normalizers(self):
        super()._build_normalizers()
        dim = self._env.get_disc_obs_space().shape[0]
        self._disc_obs_norm = normalizer.Normalizer((dim,), clip=10.0, device=self._device)
        n = 0
        while n < self._disc_norm_samples:                                   # mocap pairs only, then frozen
            batch = min(65536, self._disc_norm_samples - n)
            self._disc_obs_norm.record(self._env.fetch_disc_obs_demo(batch))
            n += batch
        self._disc_obs_norm.update()
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        dim = self._env.get_disc_obs_space().shape[0]
        self._disc_replay = ReplayBuffer(self._disc_replay_size, dim, self._device)
        return

    # ---------- rollout and rewards ----------
    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("disc_obs", next_info["disc_obs"])
        return

    def _eval_disc(self, disc_obs, batch_size=65536):
        """D on raw disc observations, by minibatches, without gradient. (n,)."""
        with torch.no_grad():
            return torch.cat([self._model.eval_disc(self._disc_obs_norm.normalize(disc_obs[i:i + batch_size]))
                              for i in range(0, disc_obs.shape[0], batch_size)])

    def _build_train_data(self):
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")               # (32 * num_envs, 210)
        task_r = self._exp_buffer.get_data_flat("reward")
        style_r = style_reward(self._eval_disc(disc_obs))
        self._exp_buffer.set_data_flat("reward", self._task_reward_weight * task_r
                                       + self._style_reward_weight * style_r)

        idx = torch.randperm(disc_obs.shape[0], device=self._device)[:self._disc_replay_insert]
        self._disc_replay.push(disc_obs[idx])

        info = super()._build_train_data()                                  # GAE on the combined reward
        info["task_reward_mean"] = task_r.mean()
        info["style_reward_mean"] = style_r.mean()
        return info

    # ---------- updates ----------
    def _update_model(self):
        info = super()._update_model()                                      # PPO: pi and V
        info.update(self._update_disc())
        return info

    def _update_disc(self):
        """disc_steps updates of D on eq. 6 + eq. 8. Returns the means of the diagnostics."""
        self._model.train()
        sums = {}
        for _ in range(self._disc_steps):
            mocap = self._disc_obs_norm.normalize(self._env.fetch_disc_obs_demo(self._disc_batch_size))
            agent = self._disc_obs_norm.normalize(self._disc_replay.sample(self._disc_batch_size))
            d_mocap = self._model.eval_disc(mocap)
            d_agent = self._model.eval_disc(agent)
            loss_ls = disc_loss_lsgan(d_mocap, d_agent)
            gp = gradient_penalty(self._model.eval_disc, mocap, self._disc_grad_penalty)
            self._disc_optimizer.step(loss_ls + gp)

            step = {"disc_loss": loss_ls, "disc_grad_penalty": gp,
                    "disc_acc_mocap": (d_mocap > 0).float().mean(),         # mocap judged mocap
                    "disc_acc_agent": (d_agent < 0).float().mean(),         # agent judged agent
                    "disc_mocap_mean": d_mocap.mean(), "disc_agent_mean": d_agent.mean()}
            for k, v in step.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
        return {k: v / self._disc_steps for k, v in sums.items()}


def build_heading_amp_agent(agent_config, env, device):
    """Called by amp/run.py when agent_name is AGENT_NAME (MimicKit's agent_builder only knows its own agents)."""
    return HeadingAMPAgent(config=agent_config, env=env, device=device)
