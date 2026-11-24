"""Plays a mocap clip through the Newton simulator and renders it to an mp4 (no display needed).

Each frame of the clip is written into Newton (pose + velocities in the convention of Newton's
eval_fk), Newton computes the bodies (eval_fk, no physics step), and we draw:

  - the skeleton as computed by Newton (blue), over the mocap skeleton (thin grey):
    they must coincide;
  - for the root, hands and feet, the velocity rebuilt from Newton's state (blue arrow,
    see check_disc_features.newton_body_origin_vel) and the velocity implied by the mocap motion
    between two frames (orange arrow): they must point the same way and have the same
    length (up to the one-frame finite difference).

    python validation/video_mocap_clip.py                          # humanoid_run
    python validation/video_mocap_clip.py --clip humanoid_walk
    python validation/video_mocap_clip.py --clip sie/humanoid_sie_run_left12 --max_frames 240
    python validation/video_mocap_clip.py --raw                    # without the velocity conversion

The velocities are written in eval_fk's convention (root center of mass, spherical joints in
the parent frame), because the arrows are computed by eval_fk. The simulator itself reads
spherical joint velocities in the child frame (see check_disc_features.py, test 1b, and check_heading_env.md, test 2).

Output: outputs/mocap/clip_<name>[_raw].mp4. Run from the project root, with .venv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display (SSH)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_disc_features as p1                        # sets paths, cwd and imports MimicKit

PROJECT = p1.PROJECT
TRACKED = ["pelvis", "right_hand", "left_hand", "right_foot", "left_foot"]
ARROW_SCALE = 0.15                         # arrow length = velocity * 0.15 s


def load_clip_frames(ml, kin, clip, max_frames, step):
    """Stored frames of one clip (no interpolation: MimicKit's slerp freezes small rotations).
    Returns the state of each kept frame, the next frame's body positions and the fps."""
    files = [str(ml.get_motion_file(i)) for i in range(ml.get_num_motions())]
    matches = [i for i, f in enumerate(files) if f.endswith(f"humanoid/{clip}.pkl")]
    if not matches:
        sys.exit(f"clip '{clip}' not in the dataset. Examples: " + ", ".join(Path(f).stem for f in files[:5]))
    m = matches[0]
    start, num = ml._motion_start_idx[m].item(), ml._motion_num_frames[m].item()
    fps = ml._motion_fps[m].item()
    idx = torch.arange(start, start + num - 1, device=p1.DEVICE)[::step][:max_frames]   # each kept frame has a next one

    state = (ml._frame_root_pos[idx], ml._frame_root_rot[idx], ml._frame_root_vel[idx],
             ml._frame_root_ang_vel[idx], ml._frame_joint_rot[idx], ml._frame_dof_vel[idx])
    bp_next, _ = kin.forward_kinematics(ml._frame_root_pos[idx + 1], ml._frame_root_rot[idx + 1],
                                        ml._frame_joint_rot[idx + 1])
    return state, bp_next, fps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default="humanoid_run", help="path under data/motions/humanoid, without .pkl")
    p.add_argument("--max_frames", type=int, default=150)
    p.add_argument("--raw", action="store_true", help="write the raw mocap velocities (no conversion)")
    args = p.parse_args()

    # Probe the clip fps to subsample 120 fps clips to ~30 fps for display.
    eng, kin, ml, cid = p1.build_env(1)
    _, _, fps = load_clip_frames(ml, kin, args.clip, 1, 1)
    step = max(1, round(fps / 30))
    n = len(range(0, ml._motion_num_frames[[str(ml.get_motion_file(i)).endswith(f"humanoid/{args.clip}.pkl")
                                            for i in range(ml.get_num_motions())].index(True)].item() - 1, step))
    n = min(n, args.max_frames)

    # One copy of the world per displayed frame: all frames are pushed to Newton at once.
    eng, kin, ml, cid = p1.build_env(n)
    state, bp_next, fps = load_clip_frames(ml, kin, args.clip, n, step)
    rp, rr, rv, ra, jr, dv = state
    if args.raw:
        rv_x, dv_x = rv, dv
    else:
        rv_x, dv_x = p1.mocap_velocities_to_newton(eng, kin, state)
        dv_x = p1.eval_fk_joint_velocities(kin, jr, dv_x)      # the arrows are computed by eval_fk
    p1.push_to_newton(eng, kin, cid, (rp, rr, rv_x, ra, jr, dv_x))

    bp_newton = eng.get_body_pos(cid)
    bv_newton = p1.newton_body_origin_vel(eng, kin, cid)  # not eng.get_body_vel: see that function
    bp_mocap, _ = kin.forward_kinematics(rp, rr, jr)
    v_mocap = (bp_next - bp_mocap) * fps                 # velocity of the body origins over one frame (drawn)
    v_exact = p1.expected_body_origin_vel(kin, state)    # exact, from the mocap velocities

    ids = [kin.get_body_id(b) for b in TRACKED]
    parents = [kin.get_parent_id(j) for j in range(bp_newton.shape[1])]
    err_pos = (bp_newton - bp_mocap).norm(dim=-1).max().item()
    err_vel = (bv_newton[:, ids] - v_exact[:, ids]).norm(dim=-1)        # (n, 5)
    err_fd = (bv_newton[:, ids] - v_mocap[:, ids]).norm(dim=-1)
    print(f"\nclip {args.clip}: {n} frames at {fps / step:.0f} fps, velocities {'raw' if args.raw else 'converted'}")
    print(f"  body positions, Newton vs mocap: max error {err_pos:.1e} m")
    for k, b in enumerate(TRACKED):
        print(f"  {b:12s} velocity, Newton vs mocap: max error {err_vel[:, k].max():.1e} m/s exact, "
              f"{err_fd[:, k].mean():.2f} m/s mean vs one-frame motion  (speed {v_mocap[:, ids[k]].norm(dim=-1).mean():.2f} m/s)")

    # ---- render ----
    P, Q, V, W = (x.cpu().numpy() for x in (bp_newton, bp_mocap, bv_newton, v_mocap))
    out = PROJECT / "outputs" / "mocap" / f"clip_{Path(args.clip).name}{'_raw' if args.raw else ''}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    writer = FFMpegWriter(fps=fps / step)
    with writer.saving(fig, str(out), dpi=100):
        for f in range(n):
            ax.cla()
            c = P[f, 0]
            ax.set_xlim(c[0] - 1, c[0] + 1); ax.set_ylim(c[1] - 1, c[1] + 1); ax.set_zlim(0, 2)
            ax.set_box_aspect((1, 1, 1))
            ax.set_title(f"{args.clip}  frame {f * step}/{n * step}  "
                         f"({'raw' if args.raw else 'converted'} velocities)", fontsize=9)
            for j in range(1, P.shape[1]):
                a = parents[j]
                ax.plot(*zip(Q[f, a], Q[f, j]), color="0.6", lw=6, alpha=0.5)    # mocap
                ax.plot(*zip(P[f, a], P[f, j]), color="tab:blue", lw=2)          # Newton
            for j in ids:
                ax.quiver(*P[f, j], *(V[f, j] * ARROW_SCALE), color="tab:blue")
                ax.quiver(*Q[f, j], *(W[f, j] * ARROW_SCALE), color="tab:orange")
            ax.text2D(0.02, 0.02, "grey/orange: mocap   blue: Newton", transform=ax.transAxes, fontsize=8)
            writer.grab_frame()
    plt.close(fig)
    print(f"  video: {out}")


if __name__ == "__main__":
    main()
