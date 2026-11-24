"""Vertical alignment of the mocap clips with the simulated ground contact.

Problem (measured by validation/check_foot_align.py): in some clips the stance feet are below the
ground. With the lowest point of the foot boxes during stance, in cm:
    simulation (Newton contact, the character's own gait)   about -1.3
    mocap Walk                                               -0.4 to -1.8
    mocap Run                                                -1.9 to -4.3
so a reference state init (RSI) from a Run clip starts with a foot 1-3 cm inside the ground,
and the contact solver pushes the character up during the first steps.

Correction: each clip m is shifted vertically by a constant
    offset_m = foot_contact_height - stance_height_m
added to the root height of all its frames (motion_lib._frame_root_pos[..., 2]). The joint
angles, the root orientation, the positions of the bodies relative to the root and all the
velocities (finite differences of a constant shift) are unchanged: only the absolute height
moves, in the RSI states and in the expert pairs of D (amp/expert.py reads the same motion lib).

stance_height_m: for each frame, lowest point of the two foot boxes (forward kinematics of the
mocap pose); the stance level is the median of the lowest STANCE_QUANTILE of the frames (the
feet are on the ground during these frames, also for running, which has flight phases).

Enabled by align_mocap_feet: True in the env config (amp/heading_env.py), off by default.
Not in the article (DeepMimic's and MimicKit's clips are used as is).
"""

from itertools import product

import torch

import util.torch_util as torch_util

FOOT_BODIES = ("right_foot", "left_foot")
# foot geometry, data/assets/humanoid/humanoid.xml: box at pos (0.045, 0, -0.0225) in the foot
# frame, half sizes (0.0885, 0.045, 0.0275)
FOOT_BOX_CENTER = (0.045, 0.0, -0.0225)
FOOT_BOX_HALF = (0.0885, 0.045, 0.0275)
STANCE_QUANTILE = 0.3


def foot_box_corners(device):
    """The 8 corners of a foot box in the foot frame, (8, 3)."""
    c = torch.tensor(FOOT_BOX_CENTER, device=device)
    h = torch.tensor(FOOT_BOX_HALF, device=device)
    return torch.stack([c + h * torch.tensor(s, device=device) for s in product([-1.0, 1.0], repeat=3)])


def lowest_foot_point(body_pos, body_rot, foot_ids):
    """Height of the lowest corner of the foot boxes. body_pos (n, B, 3), body_rot (n, B, 4)
    (world frame, quaternions xyzw as MimicKit) -> (n,)."""
    corners = foot_box_corners(body_pos.device)
    n = body_pos.shape[0]
    zs = []
    for f in foot_ids:
        q = body_rot[:, f].unsqueeze(1).expand(n, 8, 4).reshape(-1, 4)
        world = body_pos[:, f].unsqueeze(1) + torch_util.quat_rotate(q, corners.expand(n, 8, 3).reshape(-1, 3)).view(n, 8, 3)
        zs.append(world[..., 2].min(-1).values)
    return torch.stack(zs, -1).min(-1).values


def stance_height(z):
    """Stance level of a series of lowest-foot heights (frames,): median of the lowest
    STANCE_QUANTILE of the frames."""
    return z[z <= torch.quantile(z, STANCE_QUANTILE)].median()


def clip_stance_heights(motion_lib, kin_char_model, batch=8192):
    """Stance level of each clip of the motion lib, from its frames. (num_motions,)."""
    foot_ids = [kin_char_model.get_body_id(b) for b in FOOT_BODIES]
    root_pos, root_rot, joint_rot = motion_lib._frame_root_pos, motion_lib._frame_root_rot, motion_lib._frame_joint_rot
    z = torch.cat([lowest_foot_point(*kin_char_model.forward_kinematics(root_pos[i:i + batch], root_rot[i:i + batch],
                                                                        joint_rot[i:i + batch]), foot_ids)
                   for i in range(0, root_pos.shape[0], batch)])
    starts, counts = motion_lib._motion_start_idx.tolist(), motion_lib._motion_num_frames.tolist()
    return torch.stack([stance_height(z[s:s + c]) for s, c in zip(starts, counts)])


def align_clips(motion_lib, kin_char_model, foot_contact_height):
    """Shifts each clip vertically so that its stance level is foot_contact_height (m).
    Modifies the motion lib in place, returns the offsets (num_motions,) in m."""
    offsets = foot_contact_height - clip_stance_heights(motion_lib, kin_char_model)
    starts, counts = motion_lib._motion_start_idx.tolist(), motion_lib._motion_num_frames.tolist()
    for m, (s, c) in enumerate(zip(starts, counts)):
        motion_lib._frame_root_pos[s:s + c, 2] += offsets[m]
    return offsets
