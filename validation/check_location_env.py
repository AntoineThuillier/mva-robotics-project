"""Checks of the Target Location environment (amp/location_env.py, configs/location/).

Test 1: random actions: every character falls within 3 s (fall detection works).
Test 2: goal observation g = (unit direction to the target in the heading frame, distance),
        against a computation by hand (angles, not quaternions).
Test 3: task reward, against DeepMimic's formula computed step by step by hand, including the
        three special cases: within success_dist (r_vel = 1), moving away (r_vel = 0), and
        fall / too far (r_G = 0). v* by hand too: fixed, or clamp(gain * d, v_min, v_max) with
        the distance at the start of the step (tar_speed_dist_gain > 0, e.g. --env_file
        configs/location/run_env.yaml).
Test 4: target schedule: delays U(5, 10) s, distances U(0, 10) m, angles uniform.
Test 5: end of episode beyond fail_dist (15 m), with the FAIL flag.
Test 6: discriminator observations: right after a reset, the env's Φ pair equals the expert pair
        (amp/expert.py) at the same clip and times, as for Target Heading.

    python validation/check_location_env.py
    python validation/check_location_env.py --tests 2 3
    python validation/check_location_env.py --env_file configs/location/run_env.yaml

Run from the project root, with the project venv (.venv).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE

import torch

import envs.base_env as base_env
from util import torch_util as tu

import check_heading_env as che

FAIL = base_env.DoneFlags.FAIL.value


ENV_FILE = None                                       # --env_file, default configs/location/locomotion_env.yaml


def env64():
    return load_envs.location_env(64, ENV_FILE)


def report(name, ok, detail=""):
    print(f"  {name:62s} {detail}  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 1 ========
def test_random_policy():
    print("\n=== Test 1: random actions, every character falls within 3 s ===")
    env = env64()
    end_step, end_flag = che.rollout(env, che.random_actions, 90)
    t = end_step.float() / 30
    ok = bool((end_step <= 90).all() and (end_flag == FAIL).all())
    return report("falls (FAIL) within 3 s", ok, f"fall times {t.min():.2f} to {t.max():.2f} s (median {t.median():.2f})")


# ======== Test 2 ========
def test_goal_obs():
    print("\n=== Test 2: goal observation ===")
    env = env64()
    obs, _ = env.reset()
    char_id = env._get_char_id()
    root_pos, root_rot = env._engine.get_root_pos(char_id), env._engine.get_root_rot(char_id)
    g = obs[:, -3:]
    # by hand: angle of the target direction minus the character's heading
    delta = (env._tar_pos - root_pos)[:, :2]
    dist = torch.linalg.norm(delta, dim=-1)
    angle = torch.atan2(delta[:, 1], delta[:, 0]) - tu.calc_heading(root_rot)
    by_hand = torch.stack([torch.cos(angle), torch.sin(angle), dist], dim=-1)
    far = dist > 1e-3
    err = (g[far] - by_hand[far]).abs().max().item()
    ok = report("g = (cos, sin of the target angle in the heading frame, distance)", err < 1e-4, f"max error {err:.1e}")
    unit = (torch.linalg.norm(g[:, :2], dim=-1) - 1).abs().max().item()
    ok &= report("direction part has norm 1", unit < 1e-5, f"max |norm - 1| {unit:.1e}")
    return ok


# ======== Test 3 ========
def v_star_by_hand(prev_root, tar, env):
    """v* of each env, from the root-target distance at the start of the step."""
    out = []
    for i in range(prev_root.shape[0]):
        d = math.hypot(tar[i, 0].item() - prev_root[i, 0].item(), tar[i, 1].item() - prev_root[i, 1].item())
        out.append(min(max(env._tar_speed_gain * d, env._tar_speed), env._tar_speed_max)
                   if env._tar_speed_gain > 0 else env._tar_speed)
    return out


def reward_by_hand(root_pos, prev_com, com, tar, v_star, dt, env):
    """DeepMimic's SceneTargetAMP::CalcReward, written independently, one env at a time.
    v_star: list, one v* per env; k_vel = loc_vel_err_scale / v*^2."""
    out = []
    for i in range(root_pos.shape[0]):
        rx, ry = root_pos[i, 0].item() - tar[i, 0].item(), root_pos[i, 1].item() - tar[i, 1].item()
        d2 = rx * rx + ry * ry
        pos_r = math.exp(-env._loc_pos_scale * d2)
        if d2 < env._success_dist ** 2:
            vel_r = 1.0
        else:
            cx, cy = tar[i, 0].item() - com[i, 0].item(), tar[i, 1].item() - com[i, 1].item()
            n = math.hypot(cx, cy)
            cx, cy = (cx / n, cy / n) if n > 1e-4 else (0.0, 0.0)
            v = (cx * (com[i, 0] - prev_com[i, 0]).item() + cy * (com[i, 1] - prev_com[i, 1]).item()) / dt
            vs = v_star[i]
            vel_r = 0.0 if v < 0 else math.exp(-env._loc_vel_scale / vs ** 2 * max(vs - v, 0.0) ** 2)
        out.append(env._loc_pos_w * pos_r + env._loc_vel_w * vel_r)
    return torch.tensor(out, device=root_pos.device)


def test_reward():
    print("\n=== Test 3: task reward (DeepMimic's formula, by hand) ===")
    env = env64()
    n = env.get_num_envs()
    env.reset()
    char_id = env._get_char_id()
    # special cases set on purpose: a quarter of the targets right under the character
    near_ids = torch.arange(0, n, 4, device=DEVICE)
    root = env._engine.get_root_pos(char_id)
    env._tar_pos[near_ids, :2] = root[near_ids, :2] + 0.1
    err, n_near, n_away, n_fail, checked = 0.0, 0, 0, 0, 0
    v_stars = []
    for _ in range(20):
        prev_com = env._prev_com.clone()
        tar = env._tar_pos.clone()
        v_star = v_star_by_hand(env._engine.get_root_pos(char_id).clone(), tar, env)
        v_stars += v_star
        _, r, done, _ = env.step(che.mocap_targets(env))
        root, com = env._engine.get_root_pos(char_id), env.calc_com()
        expected = reward_by_hand(root, prev_com, com, tar, v_star, env._engine.get_timestep(), env)
        expected[done == FAIL] = 0.0                      # fall or too far: no task reward
        err = max(err, (r - expected).abs().max().item())
        d = torch.linalg.norm((tar - root)[:, :2], dim=-1)
        n_near += int(((d < env._success_dist) & (done != FAIL)).sum())
        n_fail += int((done == FAIL).sum())
        checked += n
        env.reset((done != 0).nonzero().flatten())
    # moving away: target placed behind the character's velocity
    env.reset()
    vel = env._ref_root_vel[:, :2]
    root = env._engine.get_root_pos(char_id)
    env._tar_pos[:, :2] = root[:, :2] - 8.0 * vel / torch.linalg.norm(vel, dim=-1, keepdim=True).clamp_min(1e-4)
    prev_com, tar = env._prev_com.clone(), env._tar_pos.clone()
    v_star = v_star_by_hand(env._engine.get_root_pos(char_id).clone(), tar, env)
    _, r, done, _ = env.step(che.mocap_targets(env))
    root, com = env._engine.get_root_pos(char_id), env.calc_com()
    expected = reward_by_hand(root, prev_com, com, tar, v_star, env._engine.get_timestep(), env)
    expected[done == FAIL] = 0.0
    err = max(err, (r - expected).abs().max().item())
    away = torch.sum(((tar - com)[:, :2]) * ((com - prev_com)[:, :2]), dim=-1) < 0
    n_away = int((away & (done != FAIL)).sum())
    ok = report("reward vs formula by hand", err < 1e-4,
                f"max error {err:.1e} on {checked + n} steps ({n_near} within 0.5 m, {n_away} moving away, {n_fail} falls)")
    ok &= report("the three special cases were exercised", n_near > 0 and n_away > 0 and n_fail > 0, "")
    vs = torch.tensor(v_stars)
    print(f"  v* over the checked steps: {vs.min():.2f} to {vs.max():.2f} m/s "
          f"(config: tar_speed {env._tar_speed}, gain {env._tar_speed_gain}, max {env._tar_speed_max})")
    if env._tar_speed_gain > 0:
        ok &= report("v* varies with the distance (min and max reached)",
                     bool(vs.min() == env._tar_speed and vs.max() == env._tar_speed_max), "")
    return ok


# ======== Test 4 ========
def test_schedule():
    print("\n=== Test 4: target schedule ===")
    env = env64()
    env.reset()
    char_id = env._get_char_id()
    delays, dists, angles = [], [], []
    for _ in range(200):
        env._reset_task(env._env_ids)
        root = env._engine.get_root_pos(char_id)
        delta = (env._tar_pos - root)[:, :2]
        delays.append(env._tar_change_times - env._time_buf)
        dists.append(torch.linalg.norm(delta, dim=-1))
        angles.append(torch.atan2(delta[:, 1], delta[:, 0]))
    delays, dists, angles = torch.cat(delays), torch.cat(dists), torch.cat(angles)
    ok = report("delays in [5, 10] s, mean ~7.5", bool(delays.min() >= 5 and delays.max() <= 10) and abs(delays.mean() - 7.5) < 0.1,
                f"{delays.min():.2f} to {delays.max():.2f}, mean {delays.mean():.3f}")
    ok &= report("distances in [0, 10] m, mean ~5", bool(dists.max() <= 10.001) and abs(dists.mean() - 5) < 0.1,
                 f"{dists.min():.2f} to {dists.max():.2f}, mean {dists.mean():.3f}")
    hist = torch.histc(angles, bins=8, min=-math.pi, max=math.pi) / angles.numel()
    ok &= report("angles uniform (8 bins, each 12.5 % +- 1 %)", bool((hist - 0.125).abs().max() < 0.01),
                 f"bins {hist.min():.3f} to {hist.max():.3f}")
    return ok


# ======== Test 5 ========
def test_fail_distance():
    print("\n=== Test 5: end of episode beyond 15 m ===")
    env = env64()
    env.reset()
    char_id = env._get_char_id()
    root = env._engine.get_root_pos(char_id)
    env._tar_pos[:32, :2] = root[:32, :2] + torch.tensor([20.0, 0.0], device=DEVICE)    # too far
    env._tar_pos[32:, :2] = root[32:, :2] + torch.tensor([3.0, 0.0], device=DEVICE)     # reachable
    _, r, done, _ = env.step(che.mocap_targets(env))
    far_fail = bool((done[:32] == FAIL).all() and (r[:32] == 0).all())
    near_ok = int((done[32:] == FAIL).sum())
    ok = report("target 20 m away: FAIL and r_G = 0", far_fail, f"{int((done[:32] == FAIL).sum())}/32")
    ok &= report("target 3 m away: no failure (except falls)", near_ok <= 2, f"{near_ok}/32 ended")
    return ok


# ======== Test 6 ========
def test_disc_obs():
    print("\n=== Test 6: discriminator observations after a reset ===")
    env = env64()
    phi, expert = env._amp_features()
    _, info = env.reset()
    dt = env._engine.get_timestep()
    ids, t = env._motion_ids.clone(), env._motion_time_offsets.clone()
    ref = phi.phi_paire(expert.state(ids, t - dt), expert.state(ids, t))
    err = (info["disc_obs"] - ref).abs().max().item()
    ok = report("env's Φ pair vs expert pair at the same clip and times", err < 1e-3, f"max error {err:.1e}")
    ok &= report("shapes (n, 210)", tuple(info["disc_obs"].shape) == (64, 210) and tuple(env.fetch_disc_obs_demo(8).shape) == (8, 210), "")
    return ok


TESTS = {
    "1": ("random actions fall", test_random_policy),
    "2": ("goal observation", test_goal_obs),
    "3": ("task reward", test_reward),
    "4": ("target schedule", test_schedule),
    "5": ("end beyond 15 m", test_fail_distance),
    "6": ("discriminator observations", test_disc_obs),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    p.add_argument("--env_file", default=None, help="env config, relative to the project root")
    args = p.parse_args()
    global ENV_FILE
    if args.env_file:
        ENV_FILE = load_envs.PROJECT / args.env_file
    results = {t: TESTS[t][1]() for t in args.tests}
    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t}  {TESTS[t][0]:35s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
