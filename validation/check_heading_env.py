"""Phase 2 validation (see CADRAGE.md, section 4, phase 2): the Target Heading environment
(amp/heading_env.py), tested with dummy policies, before any learning.

Test 1: random policy -> the character falls and the episode ends early (article, section 6.3).
Test 2: reset from the mocap (RSI): velocities written in Newton's convention, measured in the
        simulator itself (no PD, one 1/240 s step, in the air); then mocap tracking (the next
        frame of the clip is the PD target).
Test 3: task reward r_G vs eq. 11 computed by hand, and the center of mass it uses.
Test 4: goal schedule statistics vs DeepMimic (SceneHeadingAMP.cpp).
Test 5: goal observation g.
Test 6: episodes end at 20 s.

    python validation/check_heading_env.py                     # all tests, 64 envs
    python validation/check_heading_env.py --tests 3 4

Run from the project root, with the project venv (.venv).

The environments come from validation/load_envs.py and are built once per process. In a
notebook or `python -i`, the tests can be run one after the other without rebuilding them:

    import sys; sys.path.insert(0, "validation")
    import check_heading_env
    check_heading_env.test_random_policy()            # builds the environment
    check_heading_env.test_reset_and_tracking()       # reuses it (builds only the no-PD one)
"""

import argparse
import math
import sys

import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE

import torch

import envs.base_env as base_env
from util import torch_util as tu

from anim.kin_char_model import JointType

from amp.heading_env import calc_com, compute_heading_goal_obs, compute_heading_reward, root_origin_vel_to_com_vel
FAIL = base_env.DoneFlags.FAIL.value
TIME = base_env.DoneFlags.TIME.value


# ======== helpers ========
def random_actions(env):
    lo, hi = env._action_bound_low, env._action_bound_high
    return lo + (hi - lo) * torch.rand(env.get_num_envs(), lo.shape[0], device=DEVICE)


def mocap_targets(env, ahead=1):
    """PD targets = pose of each env's clip `ahead` steps later."""
    dt = env._engine.get_timestep()
    t = env._motion_time_offsets + env._time_buf + ahead * dt
    jr = env._motion_lib.calc_motion_frame(env._motion_ids, t)[4]
    return env._kin_char_model.rot_to_dof(jr)


def mocap_body_pos(env, ahead=0):
    dt = env._engine.get_timestep()
    t = env._motion_time_offsets + env._time_buf + ahead * dt
    rp, rr, _, _, jr = env._motion_lib.calc_motion_frame(env._motion_ids, t)[:5]
    return env._kin_char_model.forward_kinematics(rp, rr, jr)[0]


def rollout(env, policy, steps, on_step=None):
    """Runs `steps` steps from a reset, without resetting finished envs. Returns, per env, the
    step at which the episode ended (or steps + 1) and its flag."""
    n = env.get_num_envs()
    env.reset()
    end_step = torch.full((n,), steps + 1, device=DEVICE)
    end_flag = torch.zeros(n, dtype=torch.int, device=DEVICE)
    for k in range(1, steps + 1):
        _, _, done, _ = env.step(policy(env))
        new = (done != 0) & (end_step > steps)
        end_step[new] = k
        end_flag[new] = done[new]
        if on_step is not None:
            on_step(k, end_step > k)
    return end_step, end_flag


# ======== Test 1: random policy ========
def test_random_policy(env=None):
    """Test 1: uniform random actions within the action bounds. The character must fall:
    every episode ends with a fall within 3 s. Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    dt = env._engine.get_timestep()
    steps = int(round(3.0 / dt))
    torch.manual_seed(0)
    end_step, end_flag = rollout(env, random_actions, steps)
    fell = (end_flag == FAIL)
    t_fall = end_step[fell].float() * dt
    ok = bool(fell.all())
    print(f"\n=== Test 1: random policy, {env.get_num_envs()} episodes ===")
    print(f"  fell within 3 s: {fell.sum().item()}/{fell.numel()}  (must be all)")
    if fell.any():
        print(f"  time of the fall: min {t_fall.min().item():.2f} s, median {t_fall.median().item():.2f} s, "
              f"max {t_fall.max().item():.2f} s")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 2: reset from the mocap, and mocap tracking ========
def solver_velocity_conventions(env_nopd):
    """Writes random velocities into Newton (character 1.5 m in the air, no PD), takes one
    physics step of 1/240 s and reads what the solver did with them. Returns mean errors
    (rad/s or m/s) of the motion implied by the step vs each candidate convention."""
    eng, kin, cid = env_nopd._engine, env_nopd._kin_char_model, env_nopd._get_char_id()
    dt = eng.get_timestep()
    n = env_nopd.get_num_envs()
    ids = torch.arange(n, device=DEVICE)
    torch.manual_seed(4)
    env_nopd.reset()
    rp = eng.get_root_pos(cid).clone(); rp[:, 2] += 1.5
    rr = eng.get_root_rot(cid).clone()
    dof0 = eng.get_dof_pos(cid).clone()
    X = 3.0 * torch.randn(n, kin.get_dof_size(), device=DEVICE)
    w = 3.0 * torch.randn(n, 3, device=DEVICE)
    v = torch.randn(n, 3, device=DEVICE)
    eng.set_root_pos(ids, cid, rp); eng.set_root_rot(ids, cid, rr)
    eng.set_root_vel(ids, cid, v); eng.set_root_ang_vel(ids, cid, w)
    eng.set_dof_pos(ids, cid, dof0); eng.set_dof_vel(ids, cid, X)
    env_nopd._enable_early_termination = False
    env_nopd.step(torch.zeros_like(X))

    q0, q1 = kin.dof_to_rot(dof0), kin.dof_to_rot(eng.get_dof_pos(cid))
    dv1 = eng.get_dof_vel(cid)
    e = {"child": [], "parent": [], "read_child": [], "read_parent": []}
    for j in range(1, kin.get_num_joints()):
        if kin.get_joint(j).joint_type != JointType.SPHERICAL:
            continue
        i = kin.get_joint_dof_idx(j)
        a, b, x = q0[:, j - 1], q1[:, j - 1], X[:, i:i + 3]
        e["child"].append((tu.quat_to_exp_map(tu.quat_mul(tu.quat_conjugate(a), b)) / dt - x).norm(dim=-1).mean())
        e["parent"].append((tu.quat_to_exp_map(tu.quat_mul(b, tu.quat_conjugate(a))) / dt - x).norm(dim=-1).mean())
        e["read_child"].append((dv1[:, i:i + 3] - x).norm(dim=-1).mean())
        e["read_parent"].append((dv1[:, i:i + 3] - tu.quat_rotate(a, x)).norm(dim=-1).mean())
    e = {k: torch.stack(val).mean().item() for k, val in e.items()}

    v_origin = (eng.get_root_pos(cid) - rp)[:, :2] / dt
    v_com_case = v - torch.cross(w, tu.quat_rotate(rr, env_nopd._body_com[0].expand(n, 3)), dim=-1)
    e["root_origin"] = (v_origin - v[:, :2]).norm(dim=-1).mean().item()
    e["root_com"] = (v_origin - v_com_case[:, :2]).norm(dim=-1).mean().item()
    rr1 = eng.get_root_rot(cid)
    e["ang_world"] = (tu.quat_to_exp_map(tu.quat_mul(rr1, tu.quat_conjugate(rr))) / dt - w).norm(dim=-1).mean().item()
    e["ang_local"] = (tu.quat_to_exp_map(tu.quat_mul(tu.quat_conjugate(rr), rr1)) / dt - w).norm(dim=-1).mean().item()
    return e


def test_reset_and_tracking(env=None, env_nopd=None):
    """Test 2:
    (a) velocities written by the reset: root = mocap root velocity converted to the center of
        mass, joints = mocap joint velocities unchanged (< 1e-6);
    (b) conventions of Newton's solver (MuJoCo), measured on one 1/240 s step without PD:
        spherical joint velocities are integrated and read back in the child frame (mocap's
        convention), the root linear velocity is that of the center of mass, the root angular
        velocity is in the world frame. Criterion: the matching convention explains the motion
        within 0.5 rad/s (0.05 m/s for the root) and at least 3 times better than the other;
    (c) mocap tracking (PD target = pose of the clip one step later): mean body position error
        after one step < 3 cm. The episode length is reported, without criterion (see check_heading_env.md).
    Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    env_nopd = load_envs.heading_env_nopd() if env_nopd is None else env_nopd
    cid = env._get_char_id()
    print(f"\n=== Test 2: reset from the mocap and mocap tracking ===")

    # (a)
    torch.manual_seed(1)
    env.reset()
    rv, dv = env._engine.get_root_vel(cid), env._engine.get_dof_vel(cid)
    rv_expected = root_origin_vel_to_com_vel(env._body_com[0], env._ref_root_rot, env._ref_root_vel, env._ref_root_ang_vel)
    err_a = max((rv - rv_expected).abs().max().item(), (dv - env._ref_dof_vel).abs().max().item())
    diff_raw = (rv - env._ref_root_vel).norm(dim=-1).mean().item()
    ok_a = err_a < 1e-6
    print(f"  (a) reset: max error vs expected {err_a:.1e}; root velocity changed by the conversion: "
          f"mean {diff_raw:.3f} m/s  => {'OK' if ok_a else 'FAIL'}")

    # (b)
    e = solver_velocity_conventions(env_nopd)
    checks = [("spherical joints, integration", e["child"], e["parent"], 0.5, "child frame", "parent frame"),
              ("spherical joints, read back", e["read_child"], e["read_parent"], 0.5, "child frame", "parent frame"),
              ("root linear velocity", e["root_com"], e["root_origin"], 0.05, "center of mass", "origin"),
              ("root angular velocity", e["ang_world"], e["ang_local"], 0.5, "world frame", "root frame")]
    ok_b = True
    for name, good, other, tol, gname, oname in checks:
        ok = good < tol and 3 * good < other
        ok_b &= ok
        print(f"  (b) {name:30s} error if {gname}: {good:.3f}, if {oname}: {other:.3f}  => {'OK' if ok else 'FAIL'}")

    # (c)
    dt = env._engine.get_timestep()
    steps = int(round(5.0 / dt))
    err = {}

    def on_step(k, alive):
        if k <= 3:
            err[k] = (env._engine.get_body_pos(cid) - mocap_body_pos(env)).norm(dim=-1).mean().item()

    lengths = {}
    for name, pol in [("mocap tracking", mocap_targets), ("zero actions", lambda e_: torch.zeros_like(random_actions(e_))),
                      ("random actions", random_actions)]:
        torch.manual_seed(1)
        end_step, end_flag = rollout(env, pol, steps, on_step if name == "mocap tracking" else None)
        lengths[name] = (end_step.float().clamp(max=steps) * dt, end_flag)
    ok_c = err[1] < 0.03
    print(f"  (c) tracking, mean body position error vs mocap after 1, 2, 3 steps: "
          f"{err[1]:.3f} {err[2]:.3f} {err[3]:.3f} m  => {'OK' if ok_c else 'FAIL'}")
    for name, (t, flag) in lengths.items():
        print(f"      episode length with {name:15s} median {t.median().item():.2f} s, "
              f"max {t.max().item():.2f} s, reached 5 s: {(flag == 0).sum().item()}/{flag.numel()}")
    ok = ok_a and ok_b and ok_c
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 3: task reward ========
def kin_com(env, ml_ids, t):
    """Center of mass of the mocap pose (kinematic model + Newton's body masses)."""
    rp, rr, _, _, jr = env._motion_lib.calc_motion_frame(ml_ids, t)[:5]
    bp, br = env._kin_char_model.forward_kinematics(rp, rr, jr)
    return calc_com(bp, br, env._body_com, env._body_mass), rr


def test_task_reward(env=None):
    """Test 3:
    (a) center of mass: Newton's bodies vs the kinematic model put in Newton's state (< 1e-3 m);
    (b) eq. 11 by hand, on mocap motion: target speed v* = speed of the clip over the step,
        target direction = direction of motion turned by an angle a. Expected r_G:
        a = 0 -> 1; a = 60 deg -> exp(-0.25 (v*/2)^2); a = 180 deg -> 0; and with v* + 2 m/s
        along the motion -> exp(-1). Error < 1e-5;
    (c) facing direction as target on the running and walking clips: median r_G > 0.9;
    (d) in the environment, step by step: the reward equals eq. 11 recomputed in this test
        from Newton's bodies and the goal that was in force during the step (< 1e-4).
    Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    dt = env._engine.get_timestep()
    ml, kin = env._motion_lib, env._kin_char_model
    print(f"\n=== Test 3: task reward r_G (eq. 11) ===")

    # (a)
    env.reset()
    env.step(random_actions(env))
    cid = env._get_char_id()
    jr = kin.dof_to_rot(env._engine.get_dof_pos(cid))
    bp, br = kin.forward_kinematics(env._engine.get_root_pos(cid), env._engine.get_root_rot(cid), jr)
    err_a = (env.calc_com() - calc_com(bp, br, env._body_com, env._body_mass)).norm(dim=-1).max().item()
    ok_a = err_a < 1e-3
    print(f"  (a) COM, Newton vs kinematic model: max error {err_a:.1e} m  => {'OK' if ok_a else 'FAIL'}")

    # (b)
    n = 4096
    ids = ml.sample_motions(n)
    t = ml.sample_time(ids, truncate_time=dt)
    c0, rr = kin_com(env, ids, t)
    c1, _ = kin_com(env, ids, t + dt)
    v = (c1 - c0)[:, :2] / dt
    speed = v.norm(dim=-1)
    keep = speed > 0.3                                  # a direction of motion is defined
    c0, c1, v, speed, rr = c0[keep], c1[keep], v[keep], speed[keep], rr[keep]
    u = v / speed.unsqueeze(-1)

    def turned(a):
        ca, sa = math.cos(a), math.sin(a)
        return torch.stack([ca * u[:, 0] - sa * u[:, 1], sa * u[:, 0] + ca * u[:, 1]], dim=-1)

    cases = [("a = 0", turned(0.0), speed, torch.ones_like(speed)),
             ("a = 60 deg", turned(math.pi / 3), speed, torch.exp(-0.25 * (speed / 2) ** 2)),
             ("a = 180 deg", turned(math.pi), speed, torch.zeros_like(speed)),
             ("v* + 2 m/s", turned(0.0), speed + 2, torch.full_like(speed, math.exp(-1)))]
    ok_b = True
    for name, d, vs, expected in cases:
        r = compute_heading_reward(c0, c1, d, vs, dt, 0.25)
        e = (r - expected).abs().max().item()
        ok_b &= e < 1e-5
        print(f"  (b) {name:12s} r_G mean {r.mean().item():.3f}, max error vs hand formula {e:.1e}")
    print(f"      {keep.sum().item()} mocap transitions  => {'OK' if ok_b else 'FAIL'}")

    # (c)
    h = tu.calc_heading(rr)
    face = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
    r = compute_heading_reward(c0, c1, face, speed, dt, 0.25)
    med = r.median().item()
    ok_c = med > 0.9
    print(f"  (c) target = facing direction, v* = clip speed: median r_G {med:.3f}, "
          f"5th percentile {r.quantile(0.05).item():.3f}  => {'OK' if ok_c else 'FAIL'}")

    # (d)
    torch.manual_seed(2)
    env.reset()
    com_prev = kin_com(env, env._motion_ids, env._motion_time_offsets)[0]
    worst, n_cmp, n_upd = 0.0, 0, 0
    alive = torch.ones(env.get_num_envs(), dtype=torch.bool, device=DEVICE)
    for _ in range(60):
        h_before = env._tar_heading.clone()
        d = torch.stack([torch.cos(h_before), torch.sin(h_before)], dim=-1)
        vs = env._tar_speed.clone()
        _, rew, done, _ = env.step(mocap_targets(env))
        bpos, brot = env._engine.get_body_pos(cid), env._engine.get_body_rot(cid)
        com = calc_com(bpos, brot, env._body_com, env._body_mass)
        v_proj = ((com - com_prev)[:, :2] / dt * d).sum(-1)
        expected = torch.where(v_proj > 0, torch.exp(-0.25 * (vs - v_proj) ** 2), torch.zeros_like(v_proj))
        worst = max(worst, (rew - expected)[alive].abs().max().item() if alive.any() else 0.0)
        n_cmp += alive.sum().item()
        n_upd += ((env._tar_heading != h_before) & alive).sum().item()
        com_prev = com
        alive &= done == 0
    ok_d = worst < 1e-4
    print(f"  (d) env reward vs recomputed, {n_cmp} steps ({n_upd} with a goal update during the step): "
          f"max error {worst:.1e}  => {'OK' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 4: goal schedule ========
def test_goal_schedule(env=None, seconds=300.0):
    """Test 4: the goal update alone (no physics), over `seconds` of simulated time per env.
    Expected (DeepMimic, SceneHeadingAMP.cpp and its args file): an update every 0.2-0.5 s,
    rounded up to the 1/30 s step (mean ~0.367 s); at each update a sharp turn U(-pi, pi) with
    probability 0.01, otherwise a drift N(0, 0.15 rad); a new v* U(1, 5) with probability 0.02.
    Criteria: intervals in [0.2, 0.5 + 1/30] s, mean within 0.01 s; drift std within 3 %;
    sharp turn and speed change frequencies within 4 standard deviations; v* in [1, 5].
    Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    dt = env._engine.get_timestep()
    env.reset()
    t0 = env._time_buf.clone()
    last = env._time_buf.clone()
    intervals, deltas, dspeed = [], [], []
    for _ in range(int(seconds / dt)):
        h, v, change = env._tar_heading.clone(), env._tar_speed.clone(), env._tar_change_times.clone()
        env._time_buf += dt
        env._update_task()
        upd = env._tar_change_times != change
        intervals.append((env._time_buf - last)[upd]); last = torch.where(upd, env._time_buf, last)
        deltas.append(tu.normalize_angle(env._tar_heading - h)[upd])
        dspeed.append((env._tar_speed - v)[upd])
    env._time_buf[:] = t0
    intervals, deltas, dspeed = torch.cat(intervals), torch.cat(deltas), torch.cat(dspeed)
    m = deltas.numel()

    ok_t = intervals.min().item() >= 0.2 - 1e-4 and intervals.max().item() <= 0.5 + dt + 1e-4 \
        and abs(intervals.mean().item() - (0.35 + dt / 2)) < 0.01
    big = deltas.abs() >= 0.75                                   # 5 drift std: a sharp turn
    std = deltas[~big].std().item()
    ok_d = abs(std / 0.15 - 1) < 0.03
    p_sharp = 0.01 * (1 - 0.75 / math.pi)                        # sharp turns larger than 0.75 rad
    z_sharp = (big.sum().item() / m - p_sharp) / math.sqrt(p_sharp * (1 - p_sharp) / m)
    changed = dspeed != 0
    z_speed = (changed.float().mean().item() - 0.02) / math.sqrt(0.02 * 0.98 / m)
    vmin, vmax = env._tar_speed.min().item(), env._tar_speed.max().item()
    ok_v = vmin >= 1.0 and vmax <= 5.0
    ok = ok_t and ok_d and abs(z_sharp) < 4 and abs(z_speed) < 4 and ok_v

    print(f"\n=== Test 4: goal schedule, {env.get_num_envs()} envs x {seconds:.0f} s, {m} updates ===")
    print(f"  interval between updates: mean {intervals.mean().item():.3f} s (expected {0.35 + dt / 2:.3f}), "
          f"min {intervals.min().item():.3f}, max {intervals.max().item():.3f}  => {'OK' if ok_t else 'FAIL'}")
    print(f"  heading drift std {std:.4f} rad (expected 0.15)  => {'OK' if ok_d else 'FAIL'}")
    print(f"  sharp turns (> 0.75 rad): {big.sum().item()} = {big.float().mean().item():.4f} per update "
          f"(expected {p_sharp:.4f}), {z_sharp:+.1f} sigma  => {'OK' if abs(z_sharp) < 4 else 'FAIL'}")
    print(f"  speed changes: {changed.sum().item()} = {changed.float().mean().item():.4f} per update "
          f"(expected 0.02), {z_speed:+.1f} sigma  => {'OK' if abs(z_speed) < 4 else 'FAIL'}")
    print(f"  v* range [{vmin:.2f}, {vmax:.2f}] m/s  => {'OK' if ok_v else 'FAIL'}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 5: goal observation ========
def test_goal_obs(env=None):
    """Test 5: g = (cos D, sin D, v*), D = target heading - character heading.
    (a) random tilted root orientations: g vs the formula (< 1e-5);
    (b) facing the target -> (1, 0, v*); target 90 deg to the left -> (0, 1, v*);
    (c) in the environment, the last 3 numbers of the observation are g for Newton's root.
    Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    n = 4096
    q = tu.quat_normalize(torch.randn(n, 4, device=DEVICE))
    th = (2 * torch.rand(n, device=DEVICE) - 1) * math.pi
    vs = 1 + 4 * torch.rand(n, device=DEVICE)
    g = compute_heading_goal_obs(q, torch.stack([torch.cos(th), torch.sin(th)], -1), vs)
    D = th - tu.calc_heading(q)
    expected = torch.stack([torch.cos(D), torch.sin(D), vs], -1)
    err_a = (g - expected).abs().max().item()

    yaw = torch.tensor([0.0, 1.0, -2.0], device=DEVICE)
    qz = tu.axis_angle_to_quat(torch.tensor([[0.0, 0.0, 1.0]] * 3, device=DEVICE), yaw)
    v3 = torch.full((3,), 2.0, device=DEVICE)
    face = compute_heading_goal_obs(qz, torch.stack([torch.cos(yaw), torch.sin(yaw)], -1), v3)
    left = compute_heading_goal_obs(qz, torch.stack([torch.cos(yaw + math.pi / 2), torch.sin(yaw + math.pi / 2)], -1), v3)
    err_b = max((face - torch.tensor([1.0, 0.0, 2.0], device=DEVICE)).abs().max().item(),
                (left - torch.tensor([0.0, 1.0, 2.0], device=DEVICE)).abs().max().item())

    env.reset()
    obs, _, _, _ = env.step(random_actions(env))
    root_rot = env._engine.get_root_rot(env._get_char_id())
    err_c = (obs[:, -3:] - compute_heading_goal_obs(root_rot, env._tar_dir, env._tar_speed)).abs().max().item()

    ok = err_a < 1e-5 and err_b < 1e-5 and err_c < 1e-6
    print(f"\n=== Test 5: goal observation g ===")
    print(f"  (a) {n} random orientations: max error vs (cos D, sin D, v*) {err_a:.1e}")
    print(f"  (b) facing the target / target 90 deg left: max error {err_b:.1e}")
    print(f"  (c) observation of the env, last 3 numbers: max error {err_c:.1e}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 6: episode length ========
def test_episode_length(env=None):
    """Test 6: without early termination (the character lies on the ground), every episode ends
    with the TIME flag after 20 s = 600 steps, not before. Returns True if OK."""
    env = load_envs.heading_env() if env is None else env
    dt = env._engine.get_timestep()
    steps = int(round(env._episode_length / dt)) + 2
    env._enable_early_termination = False
    try:
        torch.manual_seed(3)
        end_step, end_flag = rollout(env, random_actions, steps)
    finally:
        env._enable_early_termination = True
    first, last = end_step.min().item(), end_step.max().item()
    ok = bool((end_flag == TIME).all()) and first == last and abs(first * dt - env._episode_length) < dt + 1e-6
    print(f"\n=== Test 6: episode length ({env._episode_length:.0f} s, early termination off) ===")
    print(f"  episodes ended at step {first}..{last} ({first * dt:.3f} s), "
          f"all with the TIME flag: {bool((end_flag == TIME).all())}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


TESTS = {
    "1": "random policy falls",
    "2": "reset velocities + mocap tracking",
    "3": "task reward r_G (eq. 11)",
    "4": "goal schedule (DeepMimic)",
    "5": "goal observation g",
    "6": "episode length 20 s",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS), help="tests to run (default: all)")
    args = p.parse_args()

    env = load_envs.heading_env(args.num_envs)
    run = {
        "1": lambda: test_random_policy(env),
        "2": lambda: test_reset_and_tracking(env),
        "3": lambda: test_task_reward(env),
        "4": lambda: test_goal_schedule(env),
        "5": lambda: test_goal_obs(env),
        "6": lambda: test_episode_length(env),
    }
    results = {t: run[t]() for t in args.tests}

    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t:3s} {TESTS[t]:45s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
