"""Evaluation of a policy trained on Target Location (rl/train.py with configs/location/), as
validation/eval_policy.py does for Target Heading.

Test 1: learning curve (log.txt of the run). Figure: <run_dir>/learning_curve.png.
Test 2: normalized task return, protocol of Table 1: 32 episodes of 20 s, mean of r_G over the
        600 steps (steps after the end count as 0). Article, Table 1: Locomotion 0.63, Zombie 0.50.
        Also the number of episodes ending with a fall, or with the target more than 15 m away.
Test 3: reaching fixed targets at 2, 4, 6, 8 and 10 m (no new target during the episode): share
        of targets reached (within 0.5 m), time to reach, and the speed towards the target as a
        function of the remaining distance. The article's Target Location behavior: walking when the
        target is near, running when it is far. Figure: <run_dir>/speed_vs_distance.png.

    python validation/eval_location.py --run_dir outputs/rl/amp_location_locomotion
    python validation/eval_location.py --run_dir outputs/rl/ppo_location --tests 2 3

Run from the project root, with the project venv (.venv).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT

import torch

import envs.base_env as base_env

import eval_policy

EP_STEPS = 600
DISTANCES = [2.0, 4.0, 6.0, 8.0, 10.0]
FAIL = base_env.DoneFlags.FAIL.value


def run_env(run_dir, num_envs):
    return load_envs.location_env(num_envs, Path(run_dir) / "env_config.yaml")


# ======== Test 2: normalized task return ========
def test_return(env, policy):
    char_id = env._get_char_id()
    ret, end_step, end_flag = eval_policy.rollout_with_reward(env, policy, EP_STEPS, seed=10)
    far = torch.linalg.norm((env._tar_pos - env._engine.get_root_pos(char_id))[:, :2], dim=-1) > env._fail_dist
    ret, end_step, end_flag, far = ret[:32], end_step[:32], end_flag[:32], far[:32]
    norm = ret / EP_STEPS
    failed = end_flag == FAIL
    n_far = int((failed & far).sum())
    print(f"\n=== Test 2: normalized task return, {norm.numel()} episodes of 20 s ===")
    print(f"  mean {norm.mean().item():.3f}, std {norm.std().item():.3f}, min {norm.min().item():.3f}, max {norm.max().item():.3f}")
    print("  article, Table 1 (AMP, Target Location): Locomotion 0.63, Zombie 0.50; Fig. 5: No Data ~0.6")
    print(f"  {int(failed.sum()) - n_far}/32 episodes end with a fall, {n_far}/32 with the target more than "
          f"{env._fail_dist:.0f} m away (mean length {end_step.clamp(max=EP_STEPS).float().mean().item() / 30:.1f} s)")
    ok = norm.mean().item() > 0.3
    print(f"  => {'OK' if ok else 'FAIL'} (sanity threshold 0.3)")
    return ok


# ======== Test 3: fixed targets ========
def test_reach(env, policy, run_dir, per_dist=12):
    """Envs in groups of per_dist, one distance of DISTANCES each, random direction; the target
    stays fixed for the whole episode. Records, while the episode lasts, the COM speed towards the
    target and the remaining distance at each step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = env.get_num_envs()
    assert n >= per_dist * len(DISTANCES)
    char_id = env._get_char_id()
    torch.manual_seed(20)
    env.reset()
    env._tar_change_times[:] = 1e9                              # no new target
    d0 = torch.tensor(DISTANCES, device=DEVICE).repeat_interleave(per_dist)
    k = d0.shape[0]
    theta = 2 * math.pi * torch.rand(k, device=DEVICE)
    root = env._engine.get_root_pos(char_id)
    env._tar_pos[:k, 0] = root[:k, 0] + d0 * torch.cos(theta)
    env._tar_pos[:k, 1] = root[:k, 1] + d0 * torch.sin(theta)
    env._obs_buf[:] = env._compute_obs()                       # the policy sees the new targets

    dt = env._engine.get_timestep()
    alive = torch.ones(n, dtype=torch.bool, device=DEVICE)
    reached_at = torch.full((n,), float("nan"), device=DEVICE)
    dists, speeds = [], []
    prev_com = env.calc_com()
    for step in range(1, EP_STEPS + 1):
        _, _, done, _ = env.step(policy(env))
        com = env.calc_com()
        to_tar = (env._tar_pos - com)[:, :2]
        dist = torch.linalg.norm((env._tar_pos - env._engine.get_root_pos(char_id))[:, :2], dim=-1)
        speed = torch.sum(to_tar / torch.linalg.norm(to_tar, dim=-1, keepdim=True).clamp_min(1e-4)
                          * (com - prev_com)[:, :2], dim=-1) / dt
        prev_com = com
        live = alive[:k]
        dists.append(dist[:k][live])
        speeds.append(speed[:k][live])
        new_reach = live & torch.isnan(reached_at[:k]) & (dist[:k] < env._success_dist)
        reached_at[:k][new_reach] = step * dt
        alive &= done == 0
    reached = ~torch.isnan(reached_at[:k])
    dists, speeds = torch.cat(dists).cpu(), torch.cat(speeds).cpu()

    print(f"\n=== Test 3: fixed targets, {per_dist} episodes per distance ===")
    print(f"  {'distance':>9s} {'reached':>9s} {'time to reach (s)':>18s}")
    for j, d in enumerate(DISTANCES):
        g = slice(j * per_dist, (j + 1) * per_dist)
        r = reached[g]
        t = reached_at[:k][g][r]
        print(f"  {d:7.0f} m {int(r.sum()):4d}/{per_dist:<4d} {('%.1f' % t.median().item()) if r.any() else '-':>18s}")
    bins = torch.arange(0, 11, 1.0)
    print("  speed towards the target vs remaining distance (median, m/s):")
    line = []
    for lo in bins[:-1]:
        m = (dists >= lo) & (dists < lo + 1)
        line.append(f"{int(lo)}-{int(lo) + 1} m: {speeds[m].median().item():.2f}" if m.sum() > 20 else f"{int(lo)}-{int(lo) + 1} m: -")
    print("    " + ", ".join(line))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(dists, speeds, s=2, alpha=0.15)
    ax.set_xlabel("distance to the target (m)")
    ax.set_ylabel("COM speed towards the target (m/s)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_title(Path(run_dir).name)
    fig.tight_layout()
    out = Path(run_dir) / "speed_vs_distance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  figure: {out}")
    ok = reached.float().mean().item() >= 0.8
    print(f"  => {'OK' if ok else 'FAIL'} ({reached.float().mean().item():.0%} of the targets reached, sanity threshold 80 %)")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="relative to the project root")
    p.add_argument("--model_file", default=None, help="default: <run_dir>/model.pt")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--tests", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"])
    args = p.parse_args()
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else PROJECT / args.run_dir

    results = {}
    if "1" in args.tests:
        results["1"] = eval_policy.test_learning_curve(run_dir)
    if "2" in args.tests or "3" in args.tests:
        env = run_env(run_dir, args.num_envs)
        policy = eval_policy.load_policy(env, run_dir, args.model_file)
        if "2" in args.tests:
            results["2"] = test_return(env, policy)
        if "3" in args.tests:
            results["3"] = test_reach(env, policy, run_dir)
    print("\n=== Summary ===")
    names = {"1": "learning curve", "2": "normalized task return", "3": "reaching fixed targets"}
    for t in sorted(results):
        print(f"  test {t}  {names[t]:30s} {'OK' if results[t] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
