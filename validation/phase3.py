"""Phase 3 validation (see CADRAGE.md, section 4, phase 3): a policy trained with PPO on the task
reward only (amp/run.py, configs/ppo_heading_humanoid_args.txt), evaluated in the Target
Heading environment. This is the "No Data" style baseline of the article (Fig. 5), except that
episodes start from mocap poses (reference state init).

Test 1: learning curve (log.txt of the run): return and episode length increase; the last
        test episodes last more than 15 s on average. Figure: <run_dir>/learning_curve.png.
Test 2: normalized task return, protocol of Table 1: 32 episodes of 20 s, mean of r_G over the
        600 steps (steps after a fall count as 0). Compared by hand to Fig. 5 ("No Data").
Test 3: speed reached vs target speed v* = 1, 2, 3, 4, 5 m/s, direction fixed (Fig. 4, right).
Test 4: falls: less than 10 % of the test episodes end with a fall.

    python validation/phase3.py                                  # run outputs/phase3/ppo_heading
    python validation/phase3.py --run_dir outputs/phase3/smoke --tests 1 2

Run from the project root, with the project venv (.venv).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT

import learning.agent_builder as agent_builder
import learning.base_agent as base_agent

import phase2

EP_STEPS = 600                                        # 20 s at 30 Hz
SPEEDS = [1.0, 2.0, 3.0, 4.0, 5.0]


# ======== helpers ========
def read_log(path):
    """MimicKit's text log: a header line, then one row of values per output iteration."""
    tokens = Path(path).read_text().split()
    ncol = next(i for i, t in enumerate(tokens) if _is_number(t))
    header, values = tokens[:ncol], tokens[ncol:]
    nrow = len(values) // ncol
    data = np.array(values[:nrow * ncol], dtype=float).reshape(nrow, ncol)
    return {k: data[:, i] for i, k in enumerate(header)}


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_policy(env, run_dir, model_file=None):
    """MimicKit agent of the run (its saved agent_config.yaml), in test mode (mean action).
    Returns policy(env) -> actions, usable with phase2.rollout."""
    run_dir = Path(run_dir)
    agent = agent_builder.build_agent(str(run_dir / "agent_config.yaml"), env, DEVICE)
    agent.load(str(model_file or run_dir / "model.pt"))
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    def policy(env_):
        with torch.no_grad():
            return agent._decide_action(env_._obs_buf, env_._info)[0]
    return policy


def rollout_with_reward(env, policy, steps, seed):
    """Like phase2.rollout, also summing the reward of each env until its episode ends."""
    n = env.get_num_envs()
    torch.manual_seed(seed)
    env.reset()
    end_step = torch.full((n,), steps + 1, device=DEVICE)
    end_flag = torch.zeros(n, dtype=torch.int, device=DEVICE)
    ret = torch.zeros(n, device=DEVICE)
    for k in range(1, steps + 1):
        _, rew, done, _ = env.step(policy(env))
        alive = end_step > steps
        ret += rew * alive.float()
        new = (done != 0) & alive
        end_step[new] = k
        end_flag[new] = done[new]
    return ret, end_step, end_flag


# ======== Test 1: learning curve ========
def test_learning_curve(run_dir):
    """Test 1: from log.txt. The test return and episode length increase between the first and
    the last output; the last test episodes last more than 15 s (450 steps) on average.
    Figure: <run_dir>/learning_curve.png. Returns True if OK."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = read_log(Path(run_dir) / "log.txt")
    s = log["Samples"] / 1e6
    ret, ep = log["Test_Return"], log["Test_Episode_Length"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(s, ret, label="test"); axes[0].plot(s, log["Train_Return"], label="train", alpha=0.6)
    axes[0].set_xlabel("samples (M)"); axes[0].set_ylabel("return (sum of r_G)"); axes[0].legend()
    axes[1].plot(s, ep / 30, label="test"); axes[1].plot(s, log["Train_Episode_Length"] / 30, label="train", alpha=0.6)
    axes[1].set_xlabel("samples (M)"); axes[1].set_ylabel("episode length (s)"); axes[1].legend()
    fig.suptitle(f"PPO, task reward only ({Path(run_dir).name})")
    fig.tight_layout()
    out = Path(run_dir) / "learning_curve.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)

    ok = len(s) >= 2 and ret[-1] > ret[0] and ep[-1] > ep[0] and ep[-1] >= 450
    print(f"\n=== Test 1: learning curve, {len(s)} outputs, {s[-1]:.1f} M samples ===")
    print(f"  test return: {ret[0]:.1f} -> {ret[-1]:.1f} (max 600)")
    print(f"  test episode length: {ep[0] / 30:.2f} s -> {ep[-1] / 30:.2f} s  (must end > 15 s)")
    print(f"  figure: {out}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Tests 2 and 4: normalized task return and falls ========
def test_return_and_falls(env, policy):
    """Test 2: normalized task return over 32 episodes of 20 s = mean of r_G over the 600 steps,
    steps after a fall counting as 0 (article, Table 1 protocol). No automatic comparison with
    the article: the value is to be compared by hand with the "No Data" curve of Fig. 5. Sanity
    criterion: > 0.5 (the character does the task most of the time).
    Test 4: less than 10 % of these episodes end with a fall.
    Returns (ok_2, ok_4)."""
    ret, end_step, end_flag = rollout_with_reward(env, policy, EP_STEPS, seed=10)
    ret, end_step, end_flag = ret[:32], end_step[:32], end_flag[:32]      # 32 episodes, as in the article
    norm = ret / EP_STEPS
    falls = (end_flag == phase2.FAIL)
    ok2 = norm.mean().item() > 0.5
    ok4 = falls.float().mean().item() < 0.1
    print(f"\n=== Test 2: normalized task return, {norm.numel()} episodes of 20 s ===")
    print(f"  mean {norm.mean().item():.3f}, std {norm.std().item():.3f}, min {norm.min().item():.3f}, max {norm.max().item():.3f}")
    print(f"  to compare by hand with Fig. 5 (No Data); article Table 1 with AMP: Locomotion 0.90")
    print(f"  => {'OK' if ok2 else 'FAIL'} (sanity threshold 0.5)")
    print(f"\n=== Test 4: falls ===")
    print(f"  {falls.sum().item()}/{falls.numel()} episodes end with a fall "
          f"(mean episode length {(end_step.clamp(max=EP_STEPS).float().mean().item() / 30):.1f} s)")
    print(f"  => {'OK' if ok4 else 'FAIL'} (must be < 10 %)")
    return ok2, ok4


# ======== Test 3: speed vs target speed ========
def test_speed_tracking(env, policy, per_speed=12):
    """Test 3: the goal schedule is frozen (no drift, turn or speed change); each group of
    `per_speed` envs gets one v* in SPEEDS. Speed along the target direction = mean COM velocity
    projected on it, from 2 s to 20 s, while the episode lasts. Criterion: within 0.5 m/s of v*
    for every v* (with at least one episode still running). Returns True if OK."""
    n = env.get_num_envs()
    assert n >= per_speed * len(SPEEDS), f"needs {per_speed * len(SPEEDS)} envs"
    dt = env._engine.get_timestep()
    saved = (env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob)
    env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob = 0.0, 0.0, 0.0
    try:
        torch.manual_seed(20)
        env.reset()
        vs = torch.full((n,), SPEEDS[-1], device=DEVICE)
        for i, v in enumerate(SPEEDS):
            vs[i * per_speed:(i + 1) * per_speed] = v
        env._tar_speed[:] = vs
        env._sync_tar_dir(env._env_ids)
        env._update_observations()                           # g in the first observation
        d = env._tar_dir.clone()
        com = env.calc_com()
        alive = torch.ones(n, dtype=torch.bool, device=DEVICE)
        v_sum = torch.zeros(n, device=DEVICE)
        cnt = torch.zeros(n, device=DEVICE)
        for k in range(1, EP_STEPS + 1):
            _, _, done, _ = env.step(policy(env))
            new_com = env.calc_com()
            if k > 60:
                m = alive.float()
                v_sum += ((new_com - com)[:, :2] / dt * d).sum(-1) * m
                cnt += m
            com = new_com
            alive &= done == 0
    finally:
        env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob = saved

    print(f"\n=== Test 3: speed along the target direction vs v* (goal frozen, {per_speed} episodes per v*) ===")
    ok = True
    for i, v in enumerate(SPEEDS):
        g = slice(i * per_speed, (i + 1) * per_speed)
        c = cnt[g].sum().item()
        reached = v_sum[g].sum().item() / c if c > 0 else float("nan")
        good = c > 0 and abs(reached - v) < 0.5
        ok &= good
        print(f"  v* = {v:.0f} m/s: reached {reached:.2f} m/s  => {'OK' if good else 'FAIL'}")
    print(f"  => {'OK' if ok else 'FAIL'} (within 0.5 m/s)")
    return ok


TESTS = {
    "1": "learning curve",
    "2": "normalized task return",
    "3": "speed vs target speed",
    "4": "falls < 10 %",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="outputs/phase3/ppo_heading", help="relative to the project root")
    p.add_argument("--model_file", default=None, help="default: <run_dir>/model.pt")
    p.add_argument("--num_envs", type=int, default=64,
                   help="evaluation envs (tests 2 and 4 use the first 32, test 3 uses 60)")
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    args = p.parse_args()
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else PROJECT / args.run_dir

    results = {}
    if "1" in args.tests:
        results["1"] = test_learning_curve(run_dir)
    if any(t in args.tests for t in ("2", "3", "4")):
        env = load_envs.heading_env(args.num_envs)
        policy = load_policy(env, run_dir, args.model_file)
        if "2" in args.tests or "4" in args.tests:
            ok2, ok4 = test_return_and_falls(env, policy)
            if "2" in args.tests:
                results["2"] = ok2
            if "4" in args.tests:
                results["4"] = ok4
        if "3" in args.tests:
            results["3"] = test_speed_tracking(env, policy)

    print("\n=== Summary ===")
    for t in sorted(results):
        print(f"  test {t:3s} {TESTS[t]:45s} {'OK' if results[t] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
