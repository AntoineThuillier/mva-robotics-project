"""Checks that our RL code (rl/) reproduces MimicKit's (mimickit/learning/), brick by brick:
same weights and same inputs must give the same outputs.

Test 1: networks: MLP, Gaussian log-probability, policy pi and value V (weights copied).
Test 2: observation normalizer (same batches recorded).
Test 3: TD(lambda) returns, with falls, time limits and the bootstrap at the end of the buffer.
Test 4: PPO on one real iteration of data (Target Heading env, 64 envs): returns and advantages,
        critic and actor losses on the same minibatch, and the parameters after one SGD step.

    python validation/rl_check.py
    python validation/rl_check.py --tests 3 4

Run from the project root, with the project venv (.venv).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT

import numpy as np
import torch
import yaml

import envs.base_env as base_env
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import learning.distribution_gaussian_diag as mk_dist
import learning.nets.fc_2layers_1024units as mk_fc
import learning.normalizer as mk_normalizer
import learning.rl_util as mk_rl_util
import util.mp_util as mp_util

from rl.models import Gaussian, GaussianPolicy, ValueFunction
from rl.nets import mlp
from rl.normalizer import Normalizer
from rl.returns import end_values, td_lambda_returns
from rl.train import Trainer

PPO_CONFIG = PROJECT / "configs" / "ppo_heading_humanoid_agent.yaml"
TOL = 1e-5


def report(name, err, tol=TOL):
    ok = err <= tol
    print(f"  {name:58s} max difference {err:.1e}  => {'OK' if ok else 'FAIL'}")
    return ok


def copy_policy_from_mimickit(mk_model, policy, value):
    """MimicKit PPOModel -> our GaussianPolicy and ValueFunction (same layer shapes)."""
    with torch.no_grad():
        policy.body.load_state_dict(mk_model._actor_layers.state_dict())
        policy.mean_out.load_state_dict(mk_model._action_dist._mean_net.state_dict())
        policy.logstd.copy_(mk_model._action_dist._logstd_net)
        value.body.load_state_dict(mk_model._critic_layers.state_dict())
        value.out.load_state_dict(mk_model._critic_out.state_dict())


def mimickit_ppo_agent(env):
    if mp_util.get_num_procs() == 0:        # MimicKit's run.py does this; without it, its means divide by 0
        mp_util.init(0, 1, DEVICE, None)
    return agent_builder.build_agent(str(PPO_CONFIG), env, DEVICE)


# ======== Test 1: networks ========
def test_networks(env=None):
    print("\n=== Test 1: networks (weights copied from MimicKit) ===")
    torch.manual_seed(0)
    ok = True
    x = torch.randn(512, 210, device=DEVICE)

    mk_net, _ = mk_fc.build_net({"x": torch.zeros(210)}, torch.nn.ReLU)
    ours = mlp(210)
    ours.load_state_dict(mk_net.state_dict())
    ok &= report("MLP 210 -> 1024 -> 512", (ours.to(DEVICE)(x) - mk_net.to(DEVICE)(x)).abs().max().item())

    mean, logstd, a = torch.randn(512, 28, device=DEVICE), torch.full((28,), np.log(0.05), device=DEVICE), torch.randn(512, 28, device=DEVICE)
    mk_lp = mk_dist.DistributionGaussianDiag(mean, torch.broadcast_to(logstd, mean.shape)).log_prob(a)
    ok &= report("Gaussian log-probability", (Gaussian(mean, logstd).log_prob(a) - mk_lp).abs().max().item(), 1e-2)
    rel = ((Gaussian(mean, logstd).log_prob(a) - mk_lp).abs() / mk_lp.abs()).max().item()
    ok &= report("  same, relative (values ~1e5, float32)", rel, 1e-6)

    env = env or load_envs.heading_env()
    mk = mimickit_ppo_agent(env)
    obs_size, a_size = env.get_obs_space().shape[0], env.get_action_space().shape[0]
    policy, value = GaussianPolicy(obs_size, a_size, 0.05).to(DEVICE), ValueFunction(obs_size).to(DEVICE)
    copy_policy_from_mimickit(mk._model, policy, value)
    obs = torch.randn(512, obs_size, device=DEVICE)
    mk_d, d = mk._model.eval_actor(obs), policy(obs)
    ok &= report("policy pi: mean of the Gaussian", (d.mean - mk_d.mean).abs().max().item())
    ok &= report("policy pi: log std", (d.logstd - mk_d.logstd).abs().max().item())
    ok &= report("value V", (value(obs) - mk._model.eval_critic(obs).squeeze(-1)).abs().max().item())
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 2: normalizer ========
def test_normalizer(env=None):
    print("\n=== Test 2: observation normalizer ===")
    torch.manual_seed(0)
    mk = mk_normalizer.Normalizer((50,), device=DEVICE, clip=10.0)
    ours = Normalizer(50, DEVICE, clip=10.0)
    for k in range(5):                                  # several updates of different sizes
        x = torch.randn(1000 * (k + 1), 50, device=DEVICE) * (k + 1) + k
        mk.record(x)
        ours.record(x)
        mk.update()
        ours.update()
    y = torch.randn(2000, 50, device=DEVICE) * 30
    ok = report("mean", (ours.mean - mk.get_mean()).abs().max().item())
    ok &= report("std", (ours.std - mk.get_std()).abs().max().item())
    ok &= report("normalize (with the +-10 clip)", (ours.normalize(y) - mk.normalize(y)).abs().max().item())
    a = Normalizer(28, DEVICE, clip=np.inf, init_mean=torch.full((28,), 0.3), init_std=torch.full((28,), 2.0))
    ok &= report("unnormalize, action normalizer", (a.unnormalize(y[:, :28]) - (y[:, :28] * 2 + 0.3)).abs().max().item())
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 3: TD(lambda) returns ========
def test_returns(env=None):
    print("\n=== Test 3: TD(lambda) returns ===")
    torch.manual_seed(0)
    T, N = 32, 256
    r = torch.rand(T, N, device=DEVICE)
    v = torch.randn(T, N, device=DEVICE) * 10
    done = torch.zeros(T, N, dtype=torch.long, device=DEVICE)
    u = torch.rand(T, N, device=DEVICE)
    done[u < 0.03] = base_env.DoneFlags.FAIL.value
    done[(u >= 0.03) & (u < 0.05)] = base_env.DoneFlags.TIME.value
    done[(u >= 0.05) & (u < 0.06)] = base_env.DoneFlags.SUCC.value

    # MimicKit: _build_train_data replaces V(s') where the episode failed or succeeded
    mk_v = v.clone()
    mk_v[done == base_env.DoneFlags.SUCC.value] = 1.5 / (1 - 0.99)
    mk_v[done == base_env.DoneFlags.FAIL.value] = -0.2 / (1 - 0.99)
    ours_v = end_values(v, done, 0.99, reward_fail=-0.2, reward_succ=1.5)
    ok = report("V(s') at the end of the episodes (fail, success)", (ours_v - mk_v).abs().max().item())
    mk_ret = mk_rl_util.compute_td_lambda_return(r, mk_v, done, 0.99, 0.95)
    ok &= report("TD(lambda) returns", (td_lambda_returns(r, ours_v, done, 0.99, 0.95) - mk_ret).abs().max().item(), 1e-4)
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 4: PPO on one real iteration ========
def test_ppo_iteration(env=None):
    print("\n=== Test 4: PPO on one real iteration (Target Heading, same data for both) ===")
    env = env or load_envs.heading_env()
    config = yaml.safe_load(open(PPO_CONFIG))
    torch.manual_seed(0)
    mk = mimickit_ppo_agent(env)
    trainer = Trainer(env, config, DEVICE)
    ours = trainer.ppo
    copy_policy_from_mimickit(mk._model, ours.policy, ours.value)

    # one rollout with our trainer (actions sampled by our policy = MimicKit's, same weights);
    # the observation normalizer is fitted on it, and copied to MimicKit
    trainer.obs, _ = env.reset()
    trainer.rollout_train()
    ours.obs_norm.update()
    with torch.no_grad():       # (MimicKit's set_mean_std fails before its first update)
        mk._obs_norm._mean.copy_(ours.obs_norm.mean)
        mk._obs_norm._std.copy_(ours.obs_norm.std)
    ok = report("action normalizer: mean", (ours.action_norm.mean - mk._a_norm.get_mean()).abs().max().item())
    ok &= report("action normalizer: std", (ours.action_norm.std - mk._a_norm.get_std()).abs().max().item())

    # the same data in MimicKit's buffer
    buf = trainer.buf
    mk._exp_buffer.reset()
    for t in range(buf.steps):
        for k_ours, k_mk in [("obs", "obs"), ("action", "action"), ("a_logp", "a_logp"), ("next_obs", "next_obs"),
                             ("reward", "reward"), ("done", "done")]:
            mk._exp_buffer.record(k_mk, buf.data[k_ours][t])
        mk._exp_buffer.record("rand_action_mask", torch.ones(buf.num_envs, device=DEVICE))
        mk._exp_buffer.inc()

    # returns and advantages
    mk.set_mode(base_agent.AgentMode.TRAIN)
    mk_info = mk._build_train_data()
    our_info = ours.compute_targets(buf, env.get_reward_fail(), env.get_reward_succ())
    ok &= report("TD(lambda) returns (tar_val)", (buf.data["tar_val"] - mk._exp_buffer.get_data("tar_val")).abs().max().item(), 1e-3)
    ok &= report("normalized advantages", (buf.data["adv"] - mk._exp_buffer.get_data("adv")).abs().max().item(), 1e-3)
    ok &= report("advantage mean and std", max(abs(float(our_info["adv_mean"] - mk_info["adv_mean"])),
                                               abs(float(our_info["adv_std"] - mk_info["adv_std"]))), 1e-3)

    # losses on the same minibatch
    idx = torch.randperm(buf.steps * buf.num_envs, device=DEVICE)[:4096]
    batch = {k: buf.flat(k)[idx] for k in buf.data}
    mk_batch = {k: mk._exp_buffer.get_data_flat(k)[idx] for k in ["obs", "action", "a_logp", "adv", "tar_val", "rand_action_mask"]}
    mk.train()
    mk_c, c = mk._compute_critic_loss(mk_batch), ours.critic_loss(batch)
    ok &= report("critic loss", abs(float(c["critic_loss"] - mk_c["critic_loss"])), 1e-3)
    mk_a, a = mk._compute_actor_loss(mk_batch), ours.actor_loss(batch)
    ok &= report("actor loss (clipped objective + action bound)", abs(float(a["actor_loss"] - mk_a["actor_loss"])), 1e-5)
    ok &= report("clip fraction", abs(float(a["clip_frac"] - mk_a["clip_frac"])))
    ok &= report("action bound loss", abs(float(a["action_bound_loss"] - mk_a.get("action_bound_loss", 0.0))))

    # one SGD step on each, from the same weights and the same minibatch
    before = ours.policy.mean_out.weight.detach().clone()
    mk._actor_optimizer.step(mk_a["actor_loss"])
    mk._critic_optimizer.step(mk_c["critic_loss"])
    for opt, loss in [(ours.actor_optimizer, a["actor_loss"]), (ours.critic_optimizer, c["critic_loss"])]:
        opt.zero_grad()
        loss.backward()
        opt.step()
    diffs = [(ours.policy.body[0].weight - mk._model._actor_layers[0].weight).abs().max().item(),
             (ours.policy.mean_out.weight - mk._model._action_dist._mean_net.weight).abs().max().item(),
             (ours.value.out.weight - mk._model._critic_out.weight).abs().max().item()]
    moved = (ours.policy.mean_out.weight - before).abs().max().item() > 0       # the step did change the weights
    ok &= report("parameters after one SGD step (pi and V)", max(diffs), 1e-6) and moved
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


TESTS = {
    "1": ("networks", test_networks),
    "2": ("observation normalizer", test_normalizer),
    "3": ("TD(lambda) returns", test_returns),
    "4": ("PPO on one real iteration", test_ppo_iteration),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    args = p.parse_args()
    results = {t: TESTS[t][1]() for t in args.tests}
    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t}  {TESTS[t][0]:35s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
