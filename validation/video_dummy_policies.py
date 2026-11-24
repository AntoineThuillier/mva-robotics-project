"""Videos of phase 2's dummy policies in the Target Heading environment (for the report).

Three mp4 files in outputs/phase2/, slowed down x0.5:

    random.mp4       uniform random actions: the character falls (test 1)
    track_walk.mp4   PD target = next frame of a walking clip (test 2)
    track_run.mp4    PD target = next frame of a running clip (test 2)

Drawn: the simulated character (blue), the mocap clip it tracks (light grey, tracking videos
only), the target direction on the ground (green arrow, length proportional to v*) and the
velocity of the center of mass (orange arrow); text: v*, speed along the target direction,
r_G, end of the episode. For the tracking videos the goal starts aligned with the clip
(direction the character faces, v* = speed of the clip), then follows the normal schedule.

    python validation/video_dummy_policies.py
    python validation/video_dummy_policies.py --videos random --seconds 2

Run from the project root, with the project venv (.venv).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                           # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT
from util import torch_util as tu

import check_heading_env

ARROW_SCALE = 0.3                          # arrow length = speed (m/s) * 0.3 s
VIDEOS = {
    # name: (policy, clip or None (random clip), start time in the clip in s)
    "random": ("random", None, 0.0),
    "track_walk": ("track", "long/humanoid_long_walk0", 10.0),
    "track_run": ("track", "humanoid_run", 0.5),
}


def find_clip(ml, clip):
    for m in range(ml.get_num_motions()):
        if str(ml.get_motion_file(m)).replace("\\", "/").endswith(f"humanoid/{clip}.pkl"):
            return m
    sys.exit(f"clip '{clip}' not in the dataset")


def reset_on(env, motion_id, t0):
    """Reset the env with the given clip and start time instead of random ones (RSI)."""
    sample = env._sample_motion_times
    env._sample_motion_times = lambda n: (torch.full((n,), motion_id, device=DEVICE, dtype=torch.long),
                                          torch.full((n,), t0, device=DEVICE))
    try:
        env.reset()
    finally:
        env._sample_motion_times = sample


def align_goal_with_clip(env):
    """Target direction = direction the character faces, v* = speed of the clip over 0.5 s."""
    ml, ids, t = env._motion_lib, env._motion_ids, env._motion_time_offsets
    p0 = ml.calc_motion_frame(ids, t)[0]
    p1 = ml.calc_motion_frame(ids, t + 0.5)[0]
    env._tar_heading[:] = tu.calc_heading(env._ref_root_rot)
    env._tar_speed[:] = ((p1 - p0)[:, :2].norm(dim=-1) / 0.5).clamp(env._tar_speed_min, env._tar_speed_max)
    env._sync_tar_dir(env._env_ids)


def record(env, policy, seconds, show_mocap):
    """Steps the env (1 copy) and records what is drawn. Keeps stepping after the end of the
    episode (no reset) to show the character on the ground."""
    dt = env._engine.get_timestep()
    cid = env._get_char_id()
    frames, end = [], None
    com = env.calc_com()
    for k in range(int(round(seconds / dt))):
        d, vs = env._tar_dir[0].clone(), env._tar_speed[0].item()
        _, rew, done, _ = env.step(policy(env))
        com_new = env.calc_com()
        v = (com_new - com)[0] / dt
        com = com_new
        if end is None and done[0].item() != 0:
            end = (k + 1, "fall" if done[0].item() == check_heading_env.FAIL else "time")
        frames.append(dict(
            t=(k + 1) * dt,
            sim=env._engine.get_body_pos(cid)[0].cpu().numpy(),
            mocap=check_heading_env.mocap_body_pos(env)[0].cpu().numpy() if show_mocap else None,
            com=com_new[0].cpu().numpy(), v=v.cpu().numpy(), d=d.cpu().numpy(), vs=vs,
            v_proj=float((d * v[:2]).sum()), r=rew[0].item(),
            end=end))
    return frames


def render(frames, parents, title, out, fps):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    fig.text(0.5, 0.03, "blue: simulated   grey: mocap   green: target direction (length v*)   "
                        "orange: COM velocity", ha="center", fontsize=8)
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=100):
        for f in frames:
            ax.cla()
            c = f["sim"][0]
            ax.set_xlim(c[0] - 1.2, c[0] + 1.2); ax.set_ylim(c[1] - 1.2, c[1] + 1.2); ax.set_zlim(0, 2.4)
            ax.set_box_aspect((1, 1, 1))
            for j in range(1, len(parents)):
                a = parents[j]
                if f["mocap"] is not None:
                    ax.plot(*zip(f["mocap"][a], f["mocap"][j]), color="0.6", lw=6, alpha=0.4)
                ax.plot(*zip(f["sim"][a], f["sim"][j]), color="tab:blue", lw=2.5)
            g = [f["com"][0], f["com"][1], 0.0]
            ax.quiver(*g, f["d"][0] * f["vs"] * ARROW_SCALE, f["d"][1] * f["vs"] * ARROW_SCALE, 0,
                      color="tab:green", lw=2.5)
            ax.quiver(*g, f["v"][0] * ARROW_SCALE, f["v"][1] * ARROW_SCALE, 0, color="tab:orange", lw=2)
            ax.set_title(title, fontsize=10)
            status = ""
            if f["end"] is not None:
                k, why = f["end"]
                status = f"   END OF EPISODE ({why}) at {k / 30:.2f} s"
            ax.text2D(0.02, 0.96, f"t = {f['t']:.2f} s{status}", transform=ax.transAxes, fontsize=9,
                      color="tab:red" if status else "black")
            ax.text2D(0.02, 0.91, f"v* = {f['vs']:.2f} m/s   speed along target = {f['v_proj']:.2f} m/s   "
                                  f"r_G = {f['r']:.2f}", transform=ax.transAxes, fontsize=9)
            writer.grab_frame()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=list(VIDEOS), choices=list(VIDEOS))
    p.add_argument("--seconds", type=float, default=3.0, help="length of each video, simulated time")
    p.add_argument("--slowdown", type=float, default=0.5, help="playback speed (0.5 = half speed)")
    args = p.parse_args()

    env = load_envs.heading_env(1)
    kin = env._kin_char_model
    parents = [kin.get_parent_id(j) for j in range(kin.get_num_joints())]
    out_dir = PROJECT / "outputs" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 30 * args.slowdown

    for name in args.videos:
        policy, clip, t0 = VIDEOS[name]
        torch.manual_seed(0)
        if clip is None:
            env.reset()
            title = "random actions"
        else:
            reset_on(env, find_clip(env._motion_lib, clip), t0)
            align_goal_with_clip(env)
            title = f"PD target = next frame of {Path(clip).name}"
        frames = record(env, check_heading_env.random_actions if policy == "random" else check_heading_env.mocap_targets,
                        args.seconds, show_mocap=clip is not None)
        out = out_dir / f"{name}.mp4"
        render(frames, parents, title, out, fps)
        end = frames[-1]["end"]
        print(f"{name}: v* at start {frames[0]['vs']:.2f} m/s,",
              f"episode ended at {end[0] / 30:.2f} s ({end[1]})" if end else "no end of episode", f"-> {out}")


if __name__ == "__main__":
    main()
