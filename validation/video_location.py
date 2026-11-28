"""Videos of a policy trained on Target Location (rl/train.py with configs/location/), as
validation/video_policy.py does for Target Heading.

Two panels: on the left, the character in 3D (blue), with the direction to the target (green
arrow, 1 m long) and the velocity of the center of mass (orange arrow); on the right, a top view
of the trajectory of the root (grey), the target (red cross) and the 0.5 m success circle.
Text: distance to the target, speed of the COM towards the target, r_G.
Two mp4 files in <run_dir>/videos/, real time:

    schedule.mp4   the normal target schedule (new target every 5 to 10 s, at 0 to 10 m)
    scripted.mp4   readable targets, placed relative to the character when they appear:
                   8 m at 45 deg to the left, then at 10 s 2 m to the right, then at 14 s
                   6 m behind (a near target should be walked to, a far one run to)

    python validation/video_location.py --run_dir outputs/rl/amp_location_locomotion
    python validation/video_location.py --run_dir outputs/rl/ppo_location --videos scripted

Run from the project root, with the project venv (.venv).
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                           # first: paths, cwd and environment cache
from load_envs import PROJECT
from util import torch_util as tu

import eval_policy
import eval_location

FAIL = eval_location.FAIL
ARROW_SCALE = 0.3                          # COM velocity arrow length = speed (m/s) * 0.3 s
SCRIPT = [  # (time in s, distance in m, angle from the direction the character faces, in rad)
    (0.0, 8.0, math.pi / 4),
    (10.0, 2.0, -math.pi / 2),
    (14.0, 6.0, math.pi),
]


def place_target(env, dist, angle):
    """Target at `dist` m and `angle` rad from the direction the character faces (env 0)."""
    cid = env._get_char_id()
    root = env._engine.get_root_pos(cid)[0]
    a = tu.calc_heading(env._engine.get_root_rot(cid))[0].item() + angle
    env._tar_pos[0, 0] = root[0] + dist * math.cos(a)
    env._tar_pos[0, 1] = root[1] + dist * math.sin(a)
    env._obs_buf[:] = env._compute_obs()      # the policy sees the new target at this step


def scripted(env, policy, script):
    """Wraps the policy: no random targets, applies `script` before each step."""
    env._tar_change_times[:] = 1e9
    done = set()

    def wrapped(env_):
        t = env_._time_buf[0].item()
        for i, (ts, d, a) in enumerate(script):
            if i not in done and t >= ts - 1e-6:
                done.add(i)
                place_target(env_, d, a)
        return policy(env_)
    return wrapped


def record(env, policy, seconds):
    """Steps the env (1 copy) and records what is drawn. Keeps stepping after the end of the
    episode (no reset) to show it."""
    dt = env._engine.get_timestep()
    cid = env._get_char_id()
    frames, end = [], None
    com = env.calc_com()
    for k in range(int(round(seconds / dt))):
        tar = env._tar_pos[0].clone()
        _, rew, done, _ = env.step(policy(env))
        com_new = env.calc_com()
        v = (com_new - com)[0] / dt
        com = com_new
        to_tar = (tar - com_new[0])[:2]
        dist = torch.linalg.norm(to_tar).item()
        d = (to_tar / max(dist, 1e-4)).cpu().numpy()
        if end is None and done[0].item() != 0:
            end = (k + 1, "fail" if done[0].item() == FAIL else "time")
        frames.append(dict(
            t=(k + 1) * dt, sim=env._engine.get_body_pos(cid)[0].cpu().numpy(),
            root=env._engine.get_root_pos(cid)[0].cpu().numpy(), com=com_new[0].cpu().numpy(),
            v=v.cpu().numpy(), d=d, tar=tar.cpu().numpy(), dist=dist,
            v_tar=float((torch.tensor(d, device=v.device) * v[:2]).sum()), r=rew[0].item(), end=end))
    return frames


def render(frames, parents, title, out, fps, success_dist):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    top = fig.add_subplot(1, 2, 2)
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.02, "blue: character   green: direction to the target   orange: COM velocity   "
                        "grey: root trajectory   red: target (circle: 0.5 m)", ha="center", fontsize=8)
    # top view limits: everything the episode visits, plus a margin, square
    pts = [f["root"][:2] for f in frames] + [f["tar"][:2] for f in frames]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys), 6.0) / 2 + 1.0
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=90):
        for i, f in enumerate(frames):
            ax.cla()
            c = f["sim"][0]
            ax.set_xlim(c[0] - 1.2, c[0] + 1.2); ax.set_ylim(c[1] - 1.2, c[1] + 1.2); ax.set_zlim(0, 2.4)
            ax.set_box_aspect((1, 1, 1))
            for j in range(1, len(parents)):
                ax.plot(*zip(f["sim"][parents[j]], f["sim"][j]), color="tab:blue", lw=2.5)
            g = [f["com"][0], f["com"][1], 0.0]
            ax.quiver(*g, f["d"][0], f["d"][1], 0, color="tab:green", lw=2.5)
            ax.quiver(*g, f["v"][0] * ARROW_SCALE, f["v"][1] * ARROW_SCALE, 0, color="tab:orange", lw=2)
            status = ""
            if f["end"] is not None:
                k, why = f["end"]
                status = f"   END OF EPISODE ({why}) at {k / 30:.2f} s"
            ax.text2D(0.0, 1.0, f"t = {f['t']:.2f} s{status}", transform=ax.transAxes, fontsize=9,
                      color="tab:red" if status else "black")
            ax.text2D(0.0, 0.95, f"distance {f['dist']:.2f} m   speed to target {f['v_tar']:.2f} m/s   "
                                 f"r_G = {f['r']:.2f}", transform=ax.transAxes, fontsize=9)

            top.cla()
            top.set_xlim(cx - half, cx + half); top.set_ylim(cy - half, cy + half); top.set_aspect("equal")
            top.grid(alpha=0.3)
            trail = [p["root"] for p in frames[:i + 1]]
            top.plot([p[0] for p in trail], [p[1] for p in trail], color="0.5", lw=1.5)
            top.plot(f["tar"][0], f["tar"][1], "x", color="tab:red", ms=12, mew=3)
            top.add_patch(plt.Circle(f["tar"][:2], success_dist, color="tab:red", fill=False, lw=1))
            top.plot(f["root"][0], f["root"][1], "o", color="tab:blue", ms=7)
            top.set_xlabel("x (m)"); top.set_ylabel("y (m)")
            writer.grab_frame()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="relative to the project root")
    p.add_argument("--model_file", default=None, help="default: <run_dir>/model.pt")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--videos", nargs="+", default=["schedule", "scripted"], choices=["schedule", "scripted"])
    args = p.parse_args()
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else PROJECT / args.run_dir

    env = eval_location.run_env(run_dir, 1)
    policy = eval_policy.load_policy(env, run_dir, args.model_file)
    kin = env._kin_char_model
    parents = [kin.get_parent_id(j) for j in range(kin.get_num_joints())]
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.videos:
        torch.manual_seed(0)
        env.reset()
        pol = scripted(env, policy, SCRIPT) if name == "scripted" else policy
        frames = record(env, pol, args.seconds)
        out = out_dir / f"{name}.mp4"
        render(frames, parents, f"{run_dir.name} ({name} targets)", out, 30, env._success_dist)
        end = frames[-1]["end"]
        print(f"{name}: " + (f"episode ended at {end[0] / 30:.2f} s ({end[1]})" if end else "no end of episode")
              + f", final distance {frames[-1]['dist']:.2f} m -> {out}")


if __name__ == "__main__":
    main()
