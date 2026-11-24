"""Validation of the vertical alignment of the mocap clips (amp/foot_align.py, align_mocap_feet).

Test 1: contact level of the simulation (lowest point of the foot boxes while on the ground),
        (a) standing still: init pose held by the PD for 1 s;
        (b) the character's own gait: policy of --run_dir (mean action), 5 s, same statistic as
            the clips (median of the lowest 30% of the frames, after the first 0.5 s).
        Prints the value to put in foot_contact_height (env config). No criterion.
Test 2: stance level of each clip, Walk and Run datasets, without and with the alignment.
        Criterion: with it, every clip at foot_contact_height within 1 mm.
Test 3: RSI, Run dataset: lowest foot point of the reset state, and vertical velocity of the
        root after the first policy step (PD target = the mocap pose one step later) minus the
        mocap's, without and with the alignment. Criterion: the upward "ejection" is reduced.
Test 4: expert pairs of D, same clips and times, without and with the alignment: Φ differs only
        by the root height (indices 0 and 105 of the pair). Criterion: other entries < 1e-4.

    python validation/check_foot_align.py                       # all tests
    python validation/check_foot_align.py --tests 2 4

Run from the project root, with the project venv (.venv).
"""

import argparse
import sys
from pathlib import Path

import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT

import torch
import yaml

from amp.foot_align import FOOT_BODIES, clip_stance_heights, lowest_foot_point, stance_height

import check_heading_env
import eval_policy

ENVS = {"Walk": PROJECT / "configs/run1/walk_env.yaml",
        "Run": PROJECT / "configs/run1/run_env.yaml"}
ALIGNED = {"Walk": PROJECT / "configs/run4/walk_env.yaml",
           "Run": PROJECT / "configs/run4/run_env.yaml"}
H, S2 = 0, 105                                        # root height in Φ(s), offset of Φ(s')


def foot_ids(env):
    kin = env._kin_char_model
    return [kin.get_body_id(b) for b in FOOT_BODIES]


def sim_lowest_foot(env):
    cid = env._get_char_id()
    return lowest_foot_point(env._engine.get_body_pos(cid), env._engine.get_body_rot(cid), foot_ids(env))


# ======== Test 1: contact level of the simulation ========
@torch.no_grad()
def test_sim_level(run_dir, num_envs):
    print("\n=== Test 1: contact level of the simulation (lowest point of the feet) ===")
    env = load_envs.heading_env(num_envs)
    cid, n = env._get_char_id(), env.get_num_envs()
    ids = torch.arange(n, device=DEVICE)

    # (a) standing: init pose, zero velocities, PD target = init pose
    env.reset()
    env._engine.set_root_pos(ids, cid, env._init_root_pos.expand(n, 3).clone())
    env._engine.set_root_rot(ids, cid, env._init_root_rot.expand(n, 4).clone())
    env._engine.set_dof_pos(ids, cid, env._init_dof_pos.expand(n, -1).clone())
    env._engine.set_root_vel(ids, cid, torch.zeros(n, 3, device=DEVICE))
    env._engine.set_root_ang_vel(ids, cid, torch.zeros(n, 3, device=DEVICE))
    env._engine.set_dof_vel(ids, cid, torch.zeros_like(env._init_dof_pos.expand(n, -1)))
    hold = env._init_dof_pos.expand(n, -1).clone()
    for _ in range(30):
        env.step(hold)
    standing = sim_lowest_foot(env).median().item()
    print(f"  (a) standing still, after 1 s: {100 * standing:+.2f} cm")

    # (b) own gait
    gait = None
    if (Path(run_dir) / "model.pt").exists():
        policy = eval_policy.load_policy(env, run_dir)
        torch.manual_seed(0)
        env.reset()
        zs = []
        for k in range(150):
            env.step(policy(env))
            if k >= 15:
                zs.append(sim_lowest_foot(env))
        z = torch.stack(zs, 0)                         # (steps, envs): stance level per env, then median
        gait = torch.stack([stance_height(z[:, i]) for i in range(n)]).median().item()
        print(f"  (b) own gait ({Path(run_dir).name}, mean action): {100 * gait:+.2f} cm")
    else:
        print(f"  (b) skipped: no {Path(run_dir) / 'model.pt'}")
    if gait is not None:
        print(f"  => foot_contact_height for this gait: {gait:.3f} m (configs: Walk {aligned_level('Walk'):.3f}, Run {aligned_level('Run'):.3f})")
    return True


def aligned_level(name):
    return float(yaml.safe_load(open(ALIGNED[name]))["foot_contact_height"])


# ======== Test 2: stance level of each clip ========
def test_clip_levels(num_envs):
    print("\n=== Test 2: stance level of each clip (cm), without / with the alignment ===")
    ok = True
    for name in ENVS:
        before = load_envs.heading_env(num_envs, ENVS[name])
        after = load_envs.heading_env(num_envs, ALIGNED[name])
        target = after._foot_contact_height
        hb = clip_stance_heights(before._motion_lib, before._kin_char_model) * 100
        ha = clip_stance_heights(after._motion_lib, after._kin_char_model) * 100
        err = (ha - 100 * target).abs().max().item()
        ok_n = err < 0.1
        ok &= ok_n
        print(f"  {name:4s} ({len(hb)} clips): without {hb.min():+.1f} to {hb.max():+.1f} (median {hb.median():+.1f}); "
              f"with {ha.min():+.2f} to {ha.max():+.2f}, target {100 * target:+.1f}, max error {10 * err:.2f} mm  "
              f"=> {'OK' if ok_n else 'FAIL'}")
    return ok


# ======== Test 3: RSI ========
@torch.no_grad()
def rsi_stats(env):
    cid = env._get_char_id()
    torch.manual_seed(1)
    env.reset()
    body_pos, body_rot = env._kin_char_model.forward_kinematics(env._ref_root_pos, env._ref_root_rot, env._ref_joint_rot)
    z0 = lowest_foot_point(body_pos, body_rot, foot_ids(env))
    dt = env._engine.get_timestep()
    t = env._motion_time_offsets + env._time_buf + dt
    env.step(check_heading_env.mocap_targets(env))
    vz_mocap = env._motion_lib.calc_motion_frame(env._motion_ids, t)[2][:, 2]
    vz = env._engine.get_root_vel(cid)[:, 2] - vz_mocap
    return z0, vz


def test_rsi(num_envs):
    print("\n=== Test 3: RSI from the Run clips, without / with the alignment ===")
    res = {}
    for tag, path in [("without", ENVS["Run"]), ("with", ALIGNED["Run"])]:
        z0, vz = rsi_stats(load_envs.heading_env(num_envs, path))
        res[tag] = vz
        print(f"  {tag:7s}: lowest foot at reset: median {100 * z0.median():+.1f} cm, below -2 cm: "
              f"{100 * (z0 < -0.02).float().mean():.0f}% | root vertical velocity - mocap after 1 step: "
              f"mean {vz.mean():+.3f}, q90 {vz.quantile(0.9):+.3f} m/s")
    ok = res["with"].quantile(0.9) < res["without"].quantile(0.9)
    print(f"  => {'OK' if ok else 'FAIL'} (upward ejection, q90, reduced)")
    return ok


# ======== Test 4: expert pairs ========
@torch.no_grad()
def test_expert_pairs(num_envs, n=4096):
    print("\n=== Test 4: expert pairs of D, without / with the alignment ===")
    ok = True
    for name in ENVS:
        before = load_envs.heading_env(num_envs, ENVS[name])
        after = load_envs.heading_env(num_envs, ALIGNED[name])
        _, ex_b = before._amp_features()
        phi_b, ids, t = ex_b.sample(n, return_info=True)
        feat_a, ex_a = after._amp_features()
        phi_a = feat_a.phi_paire(ex_a.state(ids, t), ex_a.state(ids, t + ex_a.dt))
        diff = (phi_a - phi_b).abs()
        height = torch.cat([diff[:, H], diff[:, S2 + H]])
        others = diff.clone()
        others[:, [H, S2 + H]] = 0
        expected = after.clip_height_offsets[ids].abs()
        err_h = (diff[:, H] - expected).abs().max().item()
        ok_n = others.max().item() < 1e-4 and err_h < 1e-4
        ok &= ok_n
        print(f"  {name:4s}: root height changed by {100 * height.mean():.2f} cm on average (= the clip offset, "
              f"error {err_h:.1e}); other entries: max change {others.max():.1e}  => {'OK' if ok_n else 'FAIL'}")
    return ok


TESTS = {"1": "contact level of the simulation", "2": "stance level of each clip",
         "3": "RSI from the Run clips", "4": "expert pairs of D"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--run_dir", default=str(PROJECT / "outputs/rl/amp_locomotion"), help="policy for test 1 (b)")
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    args = p.parse_args()
    args.run_dir = PROJECT / args.run_dir          # relative paths: from the project root (load_envs changes the cwd)
    run = {"1": lambda: test_sim_level(args.run_dir, args.num_envs),
           "2": lambda: test_clip_levels(args.num_envs),
           "3": lambda: test_rsi(args.num_envs),
           "4": lambda: test_expert_pairs(args.num_envs)}
    results = {t: run[t]() for t in args.tests}
    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t:3s} {TESTS[t]:35s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
