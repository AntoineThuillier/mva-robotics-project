"""Phase 1 validation (see CADRAGE.md, section 4, phase 1).

Test 1: Φ computed on the mocap == Φ recomputed after putting the Newton
simulator in the same state.
Test 1b: velocity of every body computed by Newton's eval_fk vs the one implied by the mocap:
eval_fk reads spherical joint velocities in the parent frame, unlike Newton's solver
(child frame, like the mocap). Documents why eval_fk velocities must not be used.
Test 2: content of Φ, block by block, vs DeepMimic (sizes) and MimicKit (values).
Test 3: horizontal root speed per dataset, Walk vs Run (figure for the report).
Test 4: number and duration of the clips vs the article (Table 2).
Test 5: Φ unchanged when the whole world is rotated around z or moved horizontally.
Test 6: no NaN in the mocap and in Φ.
Test 7: expert transitions sampler (amp/expert.py), end to end through Newton.

    python validation/phase1.py                     # all tests, 64 random states
    python validation/phase1.py --num_envs 256
    python validation/phase1.py --tests 3 4         # only some tests

Outputs (figures) go to outputs/phase1/.

Run from the project root, with the project venv (.venv).
"""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
MIMICKIT = PROJECT / "reference" / "MimicKit"
os.chdir(MIMICKIT)                                    # MimicKit reads its files with relative paths (data/...)
sys.path[:0] = [str(MIMICKIT / "mimickit"), str(PROJECT)]
os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / "reference" / "warpcache"))

import warp as wp

import load_envs                                      # environments shared by the validation scripts
from anim.kin_char_model import JointType
from util import torch_util as tu

from amp.disc_features import AMPFeatures, EtatPerso   # EtatPerso = character state (names in that module are French)
from amp import expert

DEVICE = "cuda:0"
TOLERANCE = 1e-3     # float32; Newton's and MimicKit's forward kinematics do not compute in the same order
VEL_TOLERANCE = 1e-2 # rad/s, float32 (measured: 3e-3 on 256 states)


# ===== build environment ======
def build_env(num_envs):
    # MimicKit's AMP steering env (data/envs/amp_steering_humanoid_env.yaml), built once per
    # process by validation/load_envs.py (see steering_env there for the build arguments).
    # num_envs = number of parallel copies of the world, here one tested state per copy.
    env = load_envs.steering_env(num_envs)
    eng = env._engine           # Newton engine, with the current state of the characters. Read with get_*, written with set_*
    kin = env._kin_char_model   # The skeleton (body tree + segments + joint types). Provides conversions and forward kinematics.
    ml = env._motion_lib        # 56 clips loaded in memory + pre-computed velocities for each frame.
    cid = env._get_char_id()    # character id (object index inside each env)
    return eng, kin, ml, cid
# =============


def sample_mocap_state(ml, n):
    # Choose a step from mocap
    ids = ml.sample_motions(n)                                  # samples n clip indexes, with the weights given in the yaml
    t   = ml.sample_time(ids)                                   # draws a random instant in each clip
    rp, rr, rv, ra, jr, dv = ml.calc_motion_frame(ids, t)[:6]   # state at these instants: root position and rotation, root linear and angular velocities, rotations of the 14 joints (quaternions) and velocities of the 28 degrees of freedom
    return ids, t, (rp, rr, rv, ra, jr, dv)


def push_to_newton(eng, kin, cid, state):
    rp, rr, rv, ra, jr, dv = state

    # Convert joint rotations to degrees of freedom
    dof = kin.rot_to_dof(jr)          # (n, 28)

    # Write the state in the engine
    env_ids = torch.arange(rp.shape[0], device=DEVICE)
    eng.set_root_pos(env_ids, cid, rp);  eng.set_root_rot(env_ids, cid, rr)
    eng.set_root_vel(env_ids, cid, rv);  eng.set_root_ang_vel(env_ids, cid, ra)
    eng.set_dof_pos(env_ids, cid, dof);  eng.set_dof_vel(env_ids, cid, dv)

    # Push the state into Newton -- otherwise Newton returns the previous pose
    s = eng._sim_state
    s.pre_step_update()    # spherical joints: exponential map -> Newton quaternions
    s.eval_fk()            # Newton computes positions and velocities of all bodies
    s.post_step_update()   # quaternions -> exponential map (tests the inverse conversion)
    torch.cuda.synchronize()


# ======== Test of our Φ (amp.disc_features) ========
def test_phi_mocap_equals_sim(eng, kin, ml, cid, n):
    """Test 1: Φ(mocap) == Φ(simulator put in the same state).

    Inputs: the Newton engine `eng`, the skeleton `kin`, the clips `ml` and the
    character id `cid` (see `build_env`), and the number of states to sample `n`
    (= num_envs: one state per copy of the world).

    Steps: sample n random instants from the mocap, put each copy of the character
    in that state, let Newton recompute the pose (body positions), then compute Φ
    on both sides, in the same heading frame.

    What it checks: the rotation <-> degree-of-freedom conversion (round trip
    through Newton's quaternions), forward kinematics (hand and foot positions)
    and our Φ applied to both sources.
    What it does NOT check: the meaning of the velocities. We read back what we
    wrote, so the velocity blocks are equal by construction (see test 1b).

    Criterion: max error on Φ < TOLERANCE. Returns True if OK.
    """
    f = AMPFeatures(kin)
    _, _, state = sample_mocap_state(ml, n)
    rp, rr, rv, ra, jr, dv = state
    push_to_newton(eng, kin, cid, state)

    bp_kin, _ = kin.forward_kinematics(rp, rr, jr)     # MimicKit forward kinematics
    s_mocap = EtatPerso(rp, rr, rv, ra, jr, dv, f.extraire_key_pos(bp_kin))
    s_sim = EtatPerso(eng.get_root_pos(cid), eng.get_root_rot(cid),
                      eng.get_root_vel(cid), eng.get_root_ang_vel(cid),
                      kin.dof_to_rot(eng.get_dof_pos(cid)), eng.get_dof_vel(cid),
                      f.extraire_key_pos(eng.get_body_pos(cid)))
    heading_inv = tu.calc_heading_quat_inv(rr)
    diff = (f(s_mocap, heading_inv) - f(s_sim, heading_inv)).abs()     # error on Φ between both sides, (n, dim)

    # Error per block of Φ (same order as AMPFeatures.__call__)
    blocks = [("root height", 1), ("root rotation", 6), ("spherical joints", 6 * len(f.ids_spheriques)),
              ("hinge joints", len(f.ids_revolutions)), ("hands and feet", 3 * len(f.key_body_ids)),
              ("root velocity", 3), ("root angular velocity", 3)]
    if f.vitesses_articulaires:
        blocks.append(("joint velocities", f.nb_dof))
    print(f"\n=== Test 1: Φ(mocap) == Φ(simulator), {n} states ===")
    start = 0
    for name, size in blocks:
        print(f"  {name:24s} max error {diff[:, start:start + size].max().item():.2e}")
        start += size
    err = diff.max().item()
    ok = err < TOLERANCE
    print(f"  => {'OK' if ok else 'FAIL'} (max error {err:.2e}, tolerance {TOLERANCE:.0e})")
    return ok


# ======== Velocity check ========
def expected_body_ang_vel(kin, state):
    """World angular velocity of every body, computed exactly from the mocap velocities
    (MimicKit conventions: joint velocities in the child frame). Returns (n, nb_bodies, 3).

    No finite difference on purpose: MimicKit's slerp (util/torch_util.py) returns the fixed
    midpoint 0.5*q0 + 0.5*q1 when two frames differ by less than 0.002 rad, which is common
    at 120 fps. The interpolated pose then freezes inside the frame and a finite difference
    reads a zero velocity."""
    rp, rr, rv, ra, jr, dv = state
    _, br = kin.forward_kinematics(rp, rr, jr)              # world rotation of every body
    w = [ra]                                                # root: world frame already
    for j in range(1, br.shape[1]):
        joint = kin.get_joint(j)
        if joint.joint_type == JointType.SPHERICAL:
            i = kin.get_joint_dof_idx(j)
            w_local = dv[:, i:i+3]
        elif joint.joint_type == JointType.HINGE:
            i = kin.get_joint_dof_idx(j)
            w_local = torch.as_tensor(joint.axis, dtype=dv.dtype, device=dv.device) * dv[:, i:i+1]
        else:                                               # FIXED (hands)
            w_local = torch.zeros_like(ra)
        # ω_child = ω_parent + (joint velocity, child frame -> world)
        w.append(w[kin.get_parent_id(j)] + tu.quat_rotate(br[:, j], w_local))
    return torch.stack(w, dim=1)


def expected_body_origin_vel(kin, state):
    """World velocity of the origin of every body, computed exactly from the mocap
    (MimicKit conventions: root velocity = velocity of the root origin). (n, nb_bodies, 3)."""
    rp, rr, rv, ra, jr, dv = state
    bp, _ = kin.forward_kinematics(rp, rr, jr)
    w = expected_body_ang_vel(kin, state)
    v = [rv]
    for j in range(1, bp.shape[1]):
        par = kin.get_parent_id(j)
        v.append(v[par] + torch.cross(w[:, par], bp[:, j] - bp[:, par], dim=-1))   # rigid transport
    return torch.stack(v, dim=1)


def newton_body_origin_vel(eng, kin, cid):
    """World velocity of the origin of every body, rebuilt from Newton's state: root velocity
    (center of mass, Newton convention), body angular velocities and positions. (n, nb_bodies, 3).

    Needed because `eng.get_body_vel` is not reliable right after eval_fk: Newton
    (sim/articulation.py, eval_single_articulation_fk) propagates the parent velocity as a
    center-of-mass velocity but stores the child's joint-anchor (origin) velocity, so the
    error builds up from the grandchildren of the root (up to ~2 m/s on hands and feet)."""
    x, q = eng.get_body_pos(cid), eng.get_body_rot(cid)
    w, v_newton = eng.get_body_ang_vel(cid), eng.get_body_vel(cid)
    n = x.shape[0]
    com = wp.to_torch(eng._sim_model.body_com).view(n, -1, 3)[:, :x.shape[1]]
    v = [v_newton[:, 0] - torch.cross(w[:, 0], tu.quat_rotate(q[:, 0], com[:, 0]), dim=-1)]   # root: COM -> origin
    for j in range(1, x.shape[1]):
        par = kin.get_parent_id(j)
        v.append(v[par] + torch.cross(w[:, par], x[:, j] - x[:, par], dim=-1))
    return torch.stack(v, dim=1)


def test_ang_vel_mocap_equals_sim(eng, kin, ml, cid, n):
    """Test 1b: body velocities computed by Newton's eval_fk vs the mocap. Documents a trap.

    Puts n random mocap states into Newton, lets eval_fk compute the velocity of every body,
    and compares it to the exact velocities implied by the mocap (`expected_body_ang_vel`,
    `expected_body_origin_vel`). Three ways of writing the velocities:
      - raw mocap velocities;
      - `mocap_velocities_to_newton` (amp/expert.py): root origin -> center of mass, joints
        unchanged. This is the convention of Newton's SOLVER (validation/phase2.py, test 2);
      - the same, plus spherical joint velocities rotated to the parent frame
        (`eval_fk_joint_velocities`). This is the convention eval_fk assumes.

    eval_fk only agrees with the mocap in the third case: eval_fk and the solver do not read
    joint_qd the same way. Phase 1 first concluded from this test that the expert velocities
    had to be rotated; the solver does not need it. Consequence: never use body velocities
    computed by eval_fk (nor `newton_body_origin_vel` after a state was written).

    Criterion: third case within VEL_TOLERANCE (rad/s and m/s), and the expert convention
    differs from eval_fk by more than 10 x VEL_TOLERANCE on the angular velocities (the trap
    is still there). Returns True if OK.
    """
    _, _, state = sample_mocap_state(ml, n)
    rp, rr, rv, ra, jr, dv = state
    w_expected = expected_body_ang_vel(kin, state)
    v_expected = expected_body_origin_vel(kin, state)
    rv_newton, dv_newton = mocap_velocities_to_newton(eng, kin, state)
    dv_fk = eval_fk_joint_velocities(kin, jr, dv_newton)

    print(f"\n=== Test 1b: body velocities, Newton's eval_fk vs mocap, {n} states ===")
    errs = {}
    for label, (rv_x, dv_x) in [("raw mocap velocities", (rv, dv)),
                                ("expert (solver) convention", (rv_newton, dv_newton)),
                                ("eval_fk convention", (rv_newton, dv_fk))]:
        push_to_newton(eng, kin, cid, (rp, rr, rv_x, ra, jr, dv_x))
        err_w = (eng.get_body_ang_vel(cid) - w_expected).norm(dim=-1).max().item()
        err_v = (newton_body_origin_vel(eng, kin, cid) - v_expected).norm(dim=-1).max().item()
        errs[label] = (err_w, err_v)
        print(f"  {label:28s} max error: angular {err_w:.2e} rad/s, linear {err_v:.2e} m/s")
    w_fk, v_fk = errs["eval_fk convention"]
    ok = w_fk < VEL_TOLERANCE and v_fk < VEL_TOLERANCE and errs["expert (solver) convention"][0] > 10 * VEL_TOLERANCE
    print(f"  eval_fk reads spherical joint velocities in the parent frame, the solver in the child frame")
    print(f"  => {'OK' if ok else 'FAIL'} (eval_fk convention within {VEL_TOLERANCE:.0e}, expert convention differs)")
    return ok


def mocap_velocities_to_newton(eng, kin, state):
    """Converts mocap velocities (MimicKit conventions) to Newton's conventions, with the
    function used for training (amp/expert.py). Returns (rv_newton, dv_newton)."""
    rp, rr, rv, ra, jr, dv = state
    return expert.mocap_velocities_to_newton(kin, expert.root_com_from_engine(eng), rr, rv, ra, jr, dv)


def eval_fk_joint_velocities(kin, joint_rot, dof_vel):
    """Spherical joint velocities rotated from the child to the parent frame (ω_parent = q · ω_child):
    the convention Newton's eval_fk assumes, NOT the one its solver uses. Only for test 1b and
    phase1_view_clip.py, which read velocities through eval_fk."""
    dof_vel = dof_vel.clone()
    for j in range(1, kin.get_num_joints()):
        if kin.get_joint(j).joint_type == JointType.SPHERICAL:
            i = kin.get_joint_dof_idx(j)
            dof_vel[..., i:i + 3] = tu.quat_rotate(joint_rot[..., j - 1, :], dof_vel[..., i:i + 3])
    return dof_vel


# ======== Helpers on the stored mocap frames ========
# The tests below read the frames stored by MotionLib directly (no interpolation: MimicKit's
# slerp freezes small rotations, see `expected_body_ang_vel`).

WALK = "walk"
RUN = "run"


def clip_dataset(path):
    """Dataset of a clip file (CADRAGE.md, section 3): Walk = humanoid_walk + long/*, Run = humanoid_run + sie/*."""
    name = Path(path).name
    if name == "humanoid_walk.pkl" or "/long/" in str(path):
        return WALK
    if name == "humanoid_run.pkl" or "/sie/" in str(path):
        return RUN
    return None


def frame_index(ml):
    """Per clip: (dataset, first frame index, number of frames, fps, length in s)."""
    clips = []
    for m in range(ml.get_num_motions()):
        clips.append((clip_dataset(ml.get_motion_file(m)), ml._motion_start_idx[m].item(),
                      ml._motion_num_frames[m].item(), ml._motion_fps[m].item(),
                      ml.get_motion_length(torch.tensor([m], device=DEVICE)).item()))
    return clips


def frame_states(ml, idx):
    """State (root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel) of the stored frames `idx`."""
    return (ml._frame_root_pos[idx], ml._frame_root_rot[idx], ml._frame_root_vel[idx],
            ml._frame_root_ang_vel[idx], ml._frame_joint_rot[idx], ml._frame_dof_vel[idx])


def to_etat(f, kin, state):
    """Mocap state -> EtatPerso (computes the hand and foot positions by forward kinematics)."""
    rp, rr, rv, ra, jr, dv = state
    bp, _ = kin.forward_kinematics(rp, rr, jr)
    return EtatPerso(rp, rr, rv, ra, jr, dv, f.extraire_key_pos(bp))


def consecutive_pairs(ml, clips, max_per_clip=None):
    """Indices (k, k+1) of consecutive stored frames, never across two clips."""
    i0 = []
    for _, start, num, _, _ in clips:
        k = torch.arange(start, start + num - 1, device=DEVICE)
        i0.append(k if max_per_clip is None else k[:: max(1, len(k) // max_per_clip)])
    i0 = torch.cat(i0)
    return i0, i0 + 1


# ======== Test 2: content of Φ vs DeepMimic and MimicKit ========
# DeepMimic, AMP observation of one state (DeepMimicCore/scenes/SceneImitateAMP.cpp,
# RecordAMPObsPose / RecordAMPObsVel; humanoid of data/characters/humanoid3d.txt:
# 8 spherical, 4 revolute, 2 fixed joints, end effectors = ankles + wrists).
# The Heading task sets --enable_amp_obs_local_root true (args/train_amp_heading_humanoid3d_locomotion_args.txt):
# root rotation and velocities in the heading frame of the NEXT state, as in our Φ.
DEEPMIMIC_BLOCKS = [            # (name, size per state, note)
    ("root height", 1, "minus ground height"),
    ("root rotation", 6, "normal then tangent (we store tangent then normal)"),
    ("joint rotations", 8 * 6 + 4 * 1, "in joint order (we group spherical, then hinges)"),
    ("end effectors", 4 * 3, "relative to the root, heading frame"),
    ("root velocity", 3, "heading frame"),
    ("root angular velocity", 3, "heading frame"),
    ("joint velocities", 8 * 4 + 4 * 1, "4 per spherical joint: 3 + one padding 0 (we store 3)"),
]
DEEPMIMIC_PAIR_ORDER = "pose(s'), pose(s), vel(s'), vel(s)   (we store Φ(s), Φ(s'))"


def phi_blocks(f):
    """Blocks of our Φ, in order: (name, size)."""
    blocks = [("root height", 1), ("root rotation", 6),
              ("joint rotations", 6 * len(f.ids_spheriques) + len(f.ids_revolutions)),
              ("end effectors", 3 * len(f.key_body_ids)), ("root velocity", 3), ("root angular velocity", 3)]
    if f.vitesses_articulaires:
        blocks.append(("joint velocities", f.nb_dof))
    return blocks


def test_phi_blocks(kin, ml, n):
    """Test 2: content and size of Φ, block by block.

    (a) Sizes compared to DeepMimic's AMP observation (table DEEPMIMIC_BLOCKS, read from
        SceneImitateAMP.cpp). The only expected size difference is the padding 0 that
        DeepMimic stores for each spherical joint velocity (8 values per state).
    (b) Values compared, on n random mocap states, to MimicKit's own observation
        (envs/char_env.py, compute_char_obs) for the blocks both compute the same way:
        everything except the hinge joints (MimicKit uses 6D, we use the angle).
        For one state, the heading frame of Φ is taken from that same state.

    Criterion: (a) sizes equal up to the padding, (b) max error < TOLERANCE. Returns True if OK.
    """
    import envs.char_env as char_env
    f = AMPFeatures(kin)
    ours = phi_blocks(f)

    print("\n=== Test 2: content of Φ, block by block ===")
    print(f"  {'block':24s} {'ours':>5s} {'DeepMimic':>9s}  note")
    ok_sizes = True
    for (name, size), (dm_name, dm_size, note) in zip(ours, DEEPMIMIC_BLOCKS):
        expected = size + (len(f.ids_spheriques) if name == "joint velocities" else 0)   # padding
        ok_sizes &= (name == dm_name and dm_size == expected)
        print(f"  {name:24s} {size:5d} {dm_size:9d}  {note}")
    print(f"  {'total per state':24s} {f.dim:5d} {sum(b[1] for b in DEEPMIMIC_BLOCKS):9d}")
    print(f"  pair order: DeepMimic {DEEPMIMIC_PAIR_ORDER}")
    print(f"  (a) sizes => {'OK' if ok_sizes else 'FAIL'}")

    # (b) numeric comparison with MimicKit's observation
    _, _, state = sample_mocap_state(ml, n)
    rp, rr, rv, ra, jr, dv = state
    e = to_etat(f, kin, state)
    phi = f(e, tu.calc_heading_quat_inv(rr))
    mk = char_env.compute_char_obs(root_pos=rp, root_rot=rr, root_vel=rv, root_ang_vel=ra, joint_rot=jr,
                                   dof_vel=dv, key_pos=e.key_pos, global_obs=False, root_height_obs=True)
    # MimicKit layout: [h, root_rot(6), root_vel(3), root_ang_vel(3), joint_rot(6 per joint), dof_vel, key_pos]
    nj = jr.shape[1]
    o_rot, o_vel = 1 + 6, 1 + 6 + 3 + 3
    mk_joint = mk[:, o_vel:o_vel + 6 * nj].reshape(n, nj, 6)
    mk_parts = {
        "root height": mk[:, 0:1],
        "root rotation": mk[:, 1:o_rot],
        "spherical joints": mk_joint[:, f.ids_spheriques].flatten(1),
        "end effectors": mk[:, o_vel + 6 * nj + f.nb_dof:],
        "root velocity": mk[:, o_rot:o_rot + 3],
        "root angular velocity": mk[:, o_rot + 3:o_vel],
        "joint velocities": mk[:, o_vel + 6 * nj:o_vel + 6 * nj + f.nb_dof],
    }
    start, ours_parts = 0, {}
    for name, size in ours:
        ours_parts[name] = phi[:, start:start + size]
        start += size
    ours_parts["spherical joints"] = ours_parts["joint rotations"][:, :6 * len(f.ids_spheriques)]
    err = 0.0
    for name, ref in mk_parts.items():
        e_b = (ours_parts[name] - ref).abs().max().item()
        err = max(err, e_b)
        print(f"  (b) {name:22s} vs MimicKit: max error {e_b:.1e}")
    ok_vals = err < TOLERANCE
    print(f"  (b) values => {'OK' if ok_vals else 'FAIL'}")
    return ok_sizes and ok_vals


# ======== Test 3: root speed distribution per dataset ========
def test_root_speed_histograms(ml, out_dir):
    """Test 3: distribution of the horizontal root speed, Walk vs Run (article, Fig. 4 and section 8.2).

    Uses every stored frame of every clip (velocity = forward difference between frames),
    saves the histogram to out_dir/root_speed_hist.png for the report.

    Criterion (CADRAGE.md): Walk around 1 m/s (median in [0.7, 1.5]); Run between 1.5 and
    5 m/s (median in [1.5, 5]). The full ranges are printed for the report. Returns True if OK.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = {WALK: [], RUN: []}
    for ds, start, num, _, _ in frame_index(ml):
        if ds is None:
            continue
        v = ml._frame_root_vel[start:start + num, :2]
        speeds[ds].append(v.norm(dim=-1))
    speeds = {ds: torch.cat(s).cpu() for ds, s in speeds.items()}

    print("\n=== Test 3: horizontal root speed per dataset (m/s) ===")
    ok = True
    criteria = {WALK: (0.7, 1.5), RUN: (1.5, 5.0)}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for ds, s in speeds.items():
        q = torch.quantile(s, torch.tensor([0.05, 0.5, 0.95]))
        lo, hi = criteria[ds]
        ok_ds = lo <= q[1].item() <= hi
        ok &= ok_ds
        print(f"  {ds:5s} {len(s):6d} frames   5%: {q[0]:.2f}   median: {q[1]:.2f}   95%: {q[2]:.2f}"
              f"   => {'OK' if ok_ds else 'FAIL'} (median in [{lo}, {hi}])")
        ax.hist(s.numpy(), bins=80, range=(0, 7), density=True, alpha=0.6, label=ds.capitalize())
    ax.set_xlabel("horizontal root speed (m/s)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "root_speed_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  figure: {out}")
    return ok


# ======== Test 4: number and duration of the clips ========
TABLE2 = {WALK: (9, 229.6), RUN: (47, 204.4)}     # article, Table 2: clips, duration (s)
DURATION_TOLERANCE = 0.5                          # s


def test_clip_counts(ml):
    """Test 4: number of clips and total duration per dataset vs the article (Table 2).

    Criterion: same number of clips, total duration within DURATION_TOLERANCE s. Returns True if OK.
    """
    print("\n=== Test 4: clips and durations vs article Table 2 ===")
    clips = frame_index(ml)
    ok = True
    for ds, (n_ref, d_ref) in TABLE2.items():
        lengths = [c[4] for c in clips if c[0] == ds]
        n, d = len(lengths), sum(lengths)
        ok_ds = n == n_ref and abs(d - d_ref) <= DURATION_TOLERANCE
        ok &= ok_ds
        print(f"  {ds:5s} {n:3d} clips, {d:6.1f} s   (article: {n_ref} clips, {d_ref} s)  => {'OK' if ok_ds else 'FAIL'}")
    others = [c for c in clips if c[0] is None]
    if others:
        ok = False
        print(f"  FAIL: {len(others)} clips belong to no dataset")
    return ok


# ======== Test 5: invariances of Φ ========
def transform_state(e, q, offset):
    """Applies a world rotation q (quaternion, (n, 4)) then a translation offset (n, 3) to an EtatPerso.
    Joint rotations and joint velocities are relative to the parent: unchanged."""
    k = e.key_pos.shape[-2]
    qk = q.unsqueeze(-2).expand(-1, k, -1)
    return EtatPerso(tu.quat_rotate(q, e.root_pos) + offset, tu.quat_mul(q, e.root_rot),
                     tu.quat_rotate(q, e.root_vel), tu.quat_rotate(q, e.root_ang_vel),
                     e.joint_rot, e.dof_vel, tu.quat_rotate(qk, e.key_pos) + offset.unsqueeze(-2))


def test_invariances(kin, ml, n):
    """Test 5: Φ of a transition does not depend on where the character is or which way it faces.

    On n pairs of consecutive mocap frames, the whole world is rotated around the vertical
    axis by a random angle and translated horizontally: phi_paire must not change.
    Negative control: a tilt (rotation around a horizontal axis) must change it, otherwise
    the test would be blind.

    Criterion: max error < TOLERANCE for the yaw + translation, > 0.1 for the tilt. Returns True if OK.
    """
    f = AMPFeatures(kin)
    i0, i1 = consecutive_pairs(ml, frame_index(ml), max_per_clip=max(1, n // ml.get_num_motions()))
    s0, s1 = to_etat(f, kin, frame_states(ml, i0)), to_etat(f, kin, frame_states(ml, i1))
    m = i0.shape[0]
    phi = f.phi_paire(s0, s1)

    z = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand(m, 3)
    yaw = tu.axis_angle_to_quat(z, (torch.rand(m, device=DEVICE) * 2 - 1) * torch.pi)
    offset = torch.zeros(m, 3, device=DEVICE)
    offset[:, :2] = (torch.rand(m, 2, device=DEVICE) * 2 - 1) * 50
    phi_moved = f.phi_paire(transform_state(s0, yaw, offset), transform_state(s1, yaw, offset))
    err = (phi - phi_moved).abs().max().item()

    x = torch.tensor([1.0, 0.0, 0.0], device=DEVICE).expand(m, 3)
    tilt = tu.axis_angle_to_quat(x, torch.full((m,), 0.5, device=DEVICE))
    zero = torch.zeros(m, 3, device=DEVICE)
    change = (phi - f.phi_paire(transform_state(s0, tilt, zero), transform_state(s1, tilt, zero))).abs().max().item()

    ok = err < TOLERANCE and change > 0.1
    print(f"\n=== Test 5: invariances of Φ, {m} transitions ===")
    print(f"  yaw rotation + horizontal translation: max change {err:.1e}  (must be < {TOLERANCE:.0e})")
    print(f"  tilt of 0.5 rad (negative control):    max change {change:.2f}  (must be > 0.1)")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 6: clean data ========
def test_no_nan(kin, ml):
    """Test 6: no NaN or infinite value, in the stored mocap frames and in Φ of every
    transition (k, k+1) of every clip (CADRAGE.md mentions NaN on the neck in humanoid_walk.pkl).

    Criterion: everything finite. Returns True if OK.
    """
    f = AMPFeatures(kin)
    clips = frame_index(ml)
    print("\n=== Test 6: no NaN in the mocap and in Φ ===")
    ok = True
    for name, x in [("root_pos", ml._frame_root_pos), ("root_rot", ml._frame_root_rot),
                    ("root_vel", ml._frame_root_vel), ("root_ang_vel", ml._frame_root_ang_vel),
                    ("joint_rot", ml._frame_joint_rot), ("dof_vel", ml._frame_dof_vel)]:
        bad = (~torch.isfinite(x)).flatten(1).any(-1)
        if bad.any():
            ok = False
            files = {Path(ml.get_motion_file(m)).name for m, (_, s, nf, _, _) in enumerate(clips)
                     if bad[s:s + nf].any()}
            print(f"  FAIL: {bad.sum().item()} frames with NaN in {name}, clips: {sorted(files)}")
    # Raw .pkl files: MimicKit turns NaN joint values into an identity rotation when loading
    # (exp map of NaN -> identity), so they never reach Φ, but the joint is frozen in that clip.
    import pickle
    import numpy as np
    for m in range(ml.get_num_motions()):
        path = MIMICKIT / ml.get_motion_file(m)
        frames = np.asarray(pickle.load(open(path, "rb"))["frames"], dtype=float)
        cols = np.where(np.isnan(frames).any(0))[0]
        if len(cols):
            print(f"  WARNING: {Path(path).name}: NaN in raw columns {cols.tolist()} "
                  f"({np.isnan(frames).any(1).sum()} frames); loaded as a frozen joint (identity rotation)")
    i0, i1 = consecutive_pairs(ml, clips)
    n_bad = 0
    for a in range(0, i0.shape[0], 8192):                 # batches, to bound memory
        b0, b1 = i0[a:a + 8192], i1[a:a + 8192]
        phi = f.phi_paire(to_etat(f, kin, frame_states(ml, b0)), to_etat(f, kin, frame_states(ml, b1)))
        n_bad += (~torch.isfinite(phi)).any(-1).sum().item()
    ok &= n_bad == 0
    print(f"  Φ on all {i0.shape[0]} transitions: {n_bad} with NaN")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 7: expert transitions (amp/expert.py) ========
def test_expert_transitions(eng, kin, ml, cid, n):
    """Test 7: the sampler of expert transitions used for training (amp/expert.py).

    (a) output shape (n, 2 * dim) and no NaN;
    (b) every pair lies inside one clip: t >= 0 and t + 1/30 s < clip length;
    (c) clips are drawn with the dataset weights (100k draws, per-clip frequency within
        4 standard deviations of the weight);
    (d) end to end: s' is put into Newton with the sampler's velocities, Φ(s') is recomputed
        from Newton's state and must equal the second half of the sampler's pair. The same
        pair with the raw (unconverted) velocities must differ, which shows the conversion
        is applied.

    Criterion: all four. Returns True if OK.
    """
    f = AMPFeatures(kin)
    sampler = expert.ExpertTransitions(ml, f, expert.root_com_from_engine(eng))
    print(f"\n=== Test 7: expert transitions (amp/expert.py), {n} pairs ===")

    phi, ids, t = sampler.sample(n, return_info=True)
    ok_a = phi.shape == (n, 2 * f.dim) and bool(torch.isfinite(phi).all())
    print(f"  (a) shape {tuple(phi.shape)}, finite: {bool(torch.isfinite(phi).all())}  => {'OK' if ok_a else 'FAIL'}")

    length = ml.get_motion_length(ids)
    ok_b = bool((t >= 0).all() and (t + sampler.dt < length).all())
    print(f"  (b) min t {t.min().item():.3f} s, min margin to clip end {(length - t - sampler.dt).min().item():.4f} s"
          f"  => {'OK' if ok_b else 'FAIL'}")

    N = 100_000
    _, ids_big, _ = sampler.sample(N, return_info=True)
    freq = torch.bincount(ids_big, minlength=ml.get_num_motions()).float() / N
    w = ml._motion_weights / ml._motion_weights.sum()
    sigma = torch.sqrt(w * (1 - w) / N)
    z = ((freq - w).abs() / sigma.clamp_min(1e-12)).max().item()
    ok_c = z < 4
    print(f"  (c) clip frequencies vs weights over {N} draws: max deviation {z:.1f} sigma  => {'OK' if ok_c else 'FAIL'}")

    # (d) end to end through Newton, for s' (the second state of each pair)
    t1 = t + sampler.dt
    rp, rr, rv, ra, jr, dv = ml.calc_motion_frame(ids, t1)[:6]
    s1 = sampler.state(ids, t1)                                      # converted velocities
    push_to_newton(eng, kin, cid, (rp, rr, s1.root_vel, ra, jr, s1.dof_vel))
    s_sim = EtatPerso(eng.get_root_pos(cid), eng.get_root_rot(cid), eng.get_root_vel(cid),
                      eng.get_root_ang_vel(cid), kin.dof_to_rot(eng.get_dof_pos(cid)), eng.get_dof_vel(cid),
                      f.extraire_key_pos(eng.get_body_pos(cid)))
    heading_inv = tu.calc_heading_quat_inv(rr)
    err = (phi[:, f.dim:] - f(s_sim, heading_inv)).abs().max().item()
    bp, _ = kin.forward_kinematics(rp, rr, jr)
    raw = f(EtatPerso(rp, rr, rv, ra, jr, dv, f.extraire_key_pos(bp)), heading_inv)
    diff_raw = (phi[:, f.dim:] - raw).abs().max().item()
    ok_d = err < TOLERANCE and diff_raw > 0.1
    print(f"  (d) Φ(s') sampler vs Newton: max error {err:.1e}; vs raw velocities: max difference {diff_raw:.2f}"
          f"  => {'OK' if ok_d else 'FAIL'}")
    ok = ok_a and ok_b and ok_c and ok_d
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


TESTS = {
    "1": "Φ(mocap) == Φ(simulator)",
    "1b": "eval_fk velocity convention (trap)",
    "2": "content of Φ vs DeepMimic and MimicKit",
    "3": "root speed per dataset (figure)",
    "4": "clips and durations vs Table 2",
    "5": "invariances of Φ",
    "6": "no NaN",
    "7": "expert transitions (amp/expert.py)",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=64, help="number of tested states (one per copy of the world)")
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS), help="tests to run (default: all)")
    args = p.parse_args()

    out_dir = PROJECT / "outputs" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    eng, kin, ml, cid = build_env(args.num_envs)
    n = args.num_envs
    run = {
        "1": lambda: test_phi_mocap_equals_sim(eng, kin, ml, cid, n),
        "1b": lambda: test_ang_vel_mocap_equals_sim(eng, kin, ml, cid, n),
        "2": lambda: test_phi_blocks(kin, ml, n),
        "3": lambda: test_root_speed_histograms(ml, out_dir),
        "4": lambda: test_clip_counts(ml),
        "5": lambda: test_invariances(kin, ml, n),
        "6": lambda: test_no_nan(kin, ml),
        "7": lambda: test_expert_transitions(eng, kin, ml, cid, n),
    }
    results = {t: run[t]() for t in args.tests}

    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t:3s} {TESTS[t]:45s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
