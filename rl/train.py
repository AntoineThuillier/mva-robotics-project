"""Training loop (replaces MimicKit's learning/base_agent.py and run.py).

    python rl/train.py --agent_config configs/run1/ppo_agent.yaml --out_dir outputs/rl/ppo_heading
    python rl/train.py --agent_config configs/run1/ppo_agent.yaml --max_samples 2e6 --out_dir outputs/rl/smoke
    python rl/train.py --mode test --model_file outputs/rl/ppo_heading/model.pt --num_envs 64
    python rl/train.py --agent_config configs/run1/amp_agent.yaml --out_dir outputs/rl/amp_heading

The agent config chooses the trainer: PPO only (Trainer below), or AMP (rl/amp.py) when it has
the discriminator settings (disc_optimizer, ...).

One iteration (as MimicKit):
  1. rollout: steps_per_iter policy steps on all environments, actions sampled from pi; an
     environment that ends (fall or time limit) is reset at once, from the mocap (RSI);
  2. TD(lambda) returns and advantages; PPO update of V then pi;
  3. during the first normalizer_samples samples, the observation normalizer takes in the
     observations of the rollout;
  4. every iters_per_output iterations: test_episodes episodes with the mean action, one line
     in log.txt (same columns and format as MimicKit, so validation/eval_policy.py reads it), model
     saved; then the training environments are reset.

Uses from MimicKit only the environment (amp/heading_env.py, built on MimicKit's envs and
Newton engine). Run from the project root, with the project venv (.venv).
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MIMICKIT = PROJECT / "reference" / "MimicKit"


def setup_paths():
    """The environments are MimicKit's: their config files use paths relative to its root."""
    os.chdir(MIMICKIT)
    # run as a script, Python puts rl/ first in sys.path, where rl/amp.py would hide the amp/ folder
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != here]
    for p in (str(PROJECT), str(MIMICKIT / "mimickit")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / "reference" / "warpcache"))


if __name__ == "__main__":
    setup_paths()

import random

import numpy as np
import torch
import yaml

from rl.buffer import RolloutBuffer
from rl.ppo import PPO
from rl.returns import DONE_NULL

ENV_MODE_TRAIN, ENV_MODE_TEST = 0, 1        # MimicKit's envs/base_env.py EnvMode


class ReturnTracker:
    """Mean return and length of the episodes finished since the last reset (MimicKit's ReturnTracker)."""

    def __init__(self, num_envs, device):
        self.ret = torch.zeros(num_envs, device=device)
        self.length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.eps_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset()

    def reset(self):
        self.episodes, self.mean_return, self.mean_length = 0, 0.0, 0.0
        self.ret.zero_()
        self.length.zero_()
        self.eps_per_env.zero_()

    def update(self, r, done):
        self.ret += r
        self.length += 1
        ids = (done != DONE_NULL).nonzero().flatten()
        if len(ids) > 0:
            n = self.episodes + len(ids)
            w_new, w_old = len(ids) / n, self.episodes / n
            self.mean_return = w_new * self.ret[ids].mean().item() + w_old * self.mean_return
            self.mean_length = w_new * self.length[ids].float().mean().item() + w_old * self.mean_length
            self.episodes = n
            self.ret[ids] = 0
            self.length[ids] = 0
            self.eps_per_env[ids] += 1


class TextLog:
    """log.txt: a header line with the column names, then one line of values per output iteration
    (MimicKit's format: fixed-width columns, lines ending with \\r)."""

    def __init__(self, path):
        self.path, self.keys = path, None

    def write(self, row):
        if self.keys is None:
            self.keys = list(row)
            with open(self.path, "w") as f:
                f.write("".join(f"{k:<25}" for k in self.keys) + "\r")
        with open(self.path, "a") as f:
            f.write("".join(f"{str(row[k]):<25}" for k in self.keys) + "\r")


class Trainer:
    """PPO training on one environment. Hooks for AMP (rl/amp.py): record_step, before_targets,
    after_update, extra_state."""

    def __init__(self, env, config, device):
        self.env, self.config, self.device = env, config, device
        self.num_envs = env.get_num_envs()
        a_space = env.get_action_space()
        self.ppo = PPO(config, env.get_obs_space().shape[0], a_space.low, a_space.high, self.num_envs, device)
        self.steps_per_iter = int(config["steps_per_iter"])
        self.iters_per_output = int(config["iters_per_output"])
        self.test_episodes = int(config["test_episodes"])
        self.normalizer_samples = float(config.get("normalizer_samples", np.inf))
        self.buf = RolloutBuffer(self.steps_per_iter, self.num_envs, device)
        self.train_tracker = ReturnTracker(self.num_envs, device)
        self.test_tracker = ReturnTracker(self.num_envs, device)
        self.sample_count = 0

    # ---------- hooks (no-op for PPO) ----------
    def record_step(self, next_info):
        """After each training step, with the env's info."""

    def before_targets(self):
        """After the rollout, before the returns and advantages (AMP: style reward)."""
        return {}

    def after_update(self):
        """After the PPO update (AMP: discriminator update)."""
        return {}

    # ---------- rollouts ----------
    def reset_done(self, done):
        ids = (done != DONE_NULL).nonzero().flatten()
        return self.env.reset(ids)

    @torch.no_grad()
    def rollout_train(self):
        self.buf.reset()
        record_obs = self.sample_count < self.normalizer_samples
        for _ in range(self.steps_per_iter):
            a, logp = self.ppo.act(self.obs, explore=True)
            self.buf.record("obs", self.obs)
            self.buf.record("action", a)
            self.buf.record("a_logp", logp)
            if record_obs:
                self.ppo.obs_norm.record(self.obs)
            next_obs, r, done, info = self.env.step(a)
            self.train_tracker.update(r, done)
            self.buf.record("next_obs", next_obs)
            self.buf.record("reward", r)
            self.buf.record("done", done)
            self.record_step(info)
            self.obs, _ = self.reset_done(done)
            self.buf.next_step()

    @torch.no_grad()
    def test(self, num_episodes):
        """Mean action, until every environment has finished ceil(num_episodes / num_envs) episodes."""
        self.ppo.eval()
        self.env.set_mode(ENV_MODE_TEST)
        self.test_tracker.reset()
        self.obs, _ = self.env.reset()
        min_eps = int(np.ceil(num_episodes / self.num_envs))
        while True:
            a, _ = self.ppo.act(self.obs, explore=False)
            self.obs, r, done, _ = self.env.step(a)
            self.test_tracker.update(r, done)
            self.obs, _ = self.reset_done(done)
            if bool(torch.all(self.test_tracker.eps_per_env >= min_eps)):
                break
        return {"mean_return": self.test_tracker.mean_return, "mean_ep_len": self.test_tracker.mean_length,
                "num_eps": self.test_tracker.episodes}

    # ---------- one iteration ----------
    def train_iter(self):
        self.ppo.eval()
        self.env.set_mode(ENV_MODE_TRAIN)
        self.rollout_train()
        info = self.before_targets()
        info.update(self.ppo.compute_targets(self.buf, self.env.get_reward_fail(), self.env.get_reward_succ()))
        info.update(self.ppo.update(self.buf))
        info.update(self.after_update())
        if self.sample_count < self.normalizer_samples:
            self.ppo.obs_norm.update()
        return info

    def train(self, max_samples, out_dir, save_int_models=True):
        out_dir = Path(out_dir)
        (out_dir / "int_models").mkdir(parents=True, exist_ok=True)
        log = TextLog(out_dir / "log.txt")
        start = time.time()
        self.obs, _ = self.env.reset()
        it, test_info = 0, None
        while self.sample_count < max_samples:
            info = self.train_iter()
            self.sample_count += self.steps_per_iter * self.num_envs
            output = it % self.iters_per_output == 0 or self.sample_count >= max_samples
            if output:
                test_info = self.test(self.test_episodes)
            wall = time.time() - start
            row = {"Iteration": it, "Wall_Time": wall / 3600, "Samples": self.sample_count,
                   "Samples_Per_Second": self.sample_count / wall,
                   "Test_Return": test_info["mean_return"], "Test_Episode_Length": test_info["mean_ep_len"],
                   "Test_Episodes": test_info["num_eps"],
                   "Train_Return": self.train_tracker.mean_return,
                   "Train_Episode_Length": self.train_tracker.mean_length,
                   "Train_Episodes": self.train_tracker.episodes}
            row.update({k.title(): float(v) for k, v in info.items()})
            row["Obs_Norm_Mean"] = self.ppo.obs_norm.mean.abs().mean().item()
            row["Obs_Norm_Std"] = self.ppo.obs_norm.std.mean().item()
            print(f"it {it:5d}  samples {self.sample_count / 1e6:7.1f} M  train return {row['Train_Return']:8.2f}"
                  f"  test return {row['Test_Return']:8.2f}  test length {row['Test_Episode_Length']:6.1f}"
                  f"  {row['Samples_Per_Second']:7.0f} samples/s", flush=True)
            if output:
                log.write(row)
                self.save(out_dir / "model.pt")
                if save_int_models:
                    self.save(out_dir / "int_models" / f"model_{it:010d}.pt")
                self.train_tracker.reset()
                self.obs, _ = self.env.reset()
            it += 1

    # ---------- saving ----------
    def extra_state(self):
        return {}

    def save(self, path):
        torch.save({"ppo": self.ppo.state_dict(), **self.extra_state()}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.ppo.load_state_dict(state["ppo"])
        return state


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(path):
    """Paths given relative to the project root (the cwd is MimicKit's root)."""
    p = Path(path)
    return str(p if p.is_absolute() or not (PROJECT / p).exists() else PROJECT / p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--agent_config", default="configs/run1/ppo_agent.yaml")
    p.add_argument("--env_config", default="configs/run1/locomotion_env.yaml")
    p.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_samples", type=float, default=3e8)
    p.add_argument("--out_dir", default="outputs/rl/ppo_heading")
    p.add_argument("--model_file", default="")
    p.add_argument("--test_episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from amp.heading_env import build_heading_env
    set_seed(args.seed)
    agent_config = resolve(args.agent_config)
    env_config = resolve(args.env_config)
    config = yaml.safe_load(open(agent_config))
    env = build_heading_env(args.num_envs, args.device, env_file=env_config, engine_file=resolve(args.engine_config))
    if "disc_optimizer" in config:                     # AMP config (configs/run1/amp_agent.yaml)
        from rl.amp import AMPTrainer
        trainer = AMPTrainer(env, config, args.device)
    else:                                              # PPO only (configs/run1/ppo_agent.yaml)
        trainer = Trainer(env, config, args.device)
    print(f"{type(trainer).__name__}: {sum(p.numel() for p in trainer.ppo.parameters() if p.requires_grad)} "
          f"parameters in pi and V", flush=True)
    if args.model_file:
        trainer.load(resolve(args.model_file))

    if args.mode == "train":
        out_dir = PROJECT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(agent_config, out_dir / "agent_config.yaml")
        shutil.copy(env_config, out_dir / "env_config.yaml")
        trainer.train(int(args.max_samples), out_dir)
    else:
        info = trainer.test(args.test_episodes)
        print(f"Mean return: {info['mean_return']:.3f}  mean episode length: {info['mean_ep_len']:.1f}"
              f"  episodes: {info['num_eps']}")


if __name__ == "__main__":
    main()
