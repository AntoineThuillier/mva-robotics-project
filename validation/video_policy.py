"""Videos of the phase 3 policy (PPO, task reward only) in the Target Heading environment.

Same drawing as validation/video_dummy_policies.py (blue: character, green: target direction with
length v*, orange: COM velocity, text: v*, speed along the target, r_G). Two mp4 files in
<run_dir>/videos/, real time:

    schedule.mp4   the normal goal schedule (DeepMimic: drift, rare turns and speed changes)
    scripted.mp4   a readable script: 2 m/s straight, then a 90 deg turn at 4 s, 4 m/s at 8 s,
                   a 135 deg turn at 12 s, 1 m/s at 16 s

With --fixed_goal, only one video instead:

    fixed_goal.mp4 a fixed goal for the whole episode: 90 deg to the left of the character, 4 m/s

    python validation/video_policy.py
    python validation/video_policy.py --fixed_goal
    python validation/video_policy.py --run_dir outputs/phase3/smoke --seconds 6
    python validation/video_policy.py --run_dir outputs/rl/amp_walk  # rl/train.py runs too

Run from the project root, with the project venv (.venv).
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                           # first: paths, cwd and environment cache
from load_envs import PROJECT
from util import torch_util as tu

import video_dummy_policies
import eval_policy

SCRIPT = [  # (time in s, heading change in rad, new v* or None)
    (0.0, 0.0, 2.0),
    (4.0, math.pi / 2, None),
    (8.0, 0.0, 4.0),
    (12.0, -3 * math.pi / 4, None),
    (16.0, 0.0, 1.0),
]
FIXED_SCRIPT = [(0.0, math.pi / 2, 4.0)]


def scripted(env, policy, script):
    """Wraps the policy: freezes the goal schedule and applies `script` before each step."""
    env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob = 0.0, 0.0, 0.0
    done = set()

    def wrapped(env_):
        t = env_._time_buf[0].item()
        for i, (ts, dh, v) in enumerate(script):
            if i not in done and t >= ts - 1e-6:
                done.add(i)
                env_._tar_heading += dh
                if v is not None:
                    env_._tar_speed[:] = v
                env_._sync_tar_dir(env_._env_ids)
                env_._update_observations()      # the policy sees the new goal at this step
        return policy(env_)
    return wrapped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="outputs/phase3/ppo_heading", help="relative to the project root")
    p.add_argument("--model_file", default=None, help="default: <run_dir>/model.pt")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fixed_goal", action="store_true", help="only fixed_goal.mp4 (FIXED_SCRIPT)")
    args = p.parse_args()
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else PROJECT / args.run_dir

    env = eval_policy.run_env(run_dir, 1)
    policy = eval_policy.load_policy(env, run_dir, args.model_file)
    kin = env._kin_char_model
    parents = [kin.get_parent_id(j) for j in range(kin.get_num_joints())]
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = (env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob)

    scripts = {"fixed_goal": FIXED_SCRIPT} if args.fixed_goal else {"schedule": None, "scripted": SCRIPT}
    for name, script in scripts.items():
        torch.manual_seed(0)
        env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob = saved
        env.reset()
        pol = policy
        if script is not None:
            env._tar_heading[:] = tu.calc_heading(env._ref_root_rot)     # start: the way the character faces
            pol = scripted(env, policy, script)
        frames = video_dummy_policies.record(env, pol, args.seconds, show_mocap=False)
        out = out_dir / f"{name}.mp4"
        video_dummy_policies.render(frames, parents, f"{run_dir.name} ({name} goal)", out, fps=30)
        end = frames[-1]["end"]
        print(f"{name}: " + (f"episode ended at {end[0] / 30:.2f} s ({end[1]})" if end else "no fall") + f" -> {out}")
    env._heading_drift_std, env._sharp_turn_prob, env._speed_change_prob = saved


if __name__ == "__main__":
    main()
