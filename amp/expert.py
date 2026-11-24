"""Expert ("real") transitions for the AMP discriminator (phase 1, CADRAGE.md).

    sampler = ExpertTransitions(motion_lib, features, root_com_from_engine(engine))
    phi = sampler.sample(4096)                 # (4096, 2 * features.dim), the discriminator's +1 inputs

A transition is (s, s') with s' = s + 1/30 s, taken in one clip at a continuous random
instant (frames are interpolated by MimicKit's MotionLib). Both states go through Φ
(`AMPFeatures.phi_paire`, heading frame of s').

Velocities are converted to Newton's conventions first (`mocap_velocities_to_newton`):
the agent's states are read from Newton, so the expert states must speak the same
language, otherwise the discriminator could tell them apart from the velocity
convention alone. Only the root linear velocity differs (validation/check_heading_env.md, test 2).

Like disc_features.py, import with MimicKit's `mimickit/` folder in sys.path.
"""

import torch
import warp as wp

import util.torch_util as torch_util

from amp.disc_features import EtatPerso

CONTROL_DT = 1.0 / 30.0        # time between s and s' (policy frequency)


def root_com_from_engine(engine, char_id=0):
    """Center of mass of the root body, in the root frame (3,), as Newton uses it (≈ (0, 0, 0.113)).
    Assumes the character's bodies come first in each world, as in MimicKit's envs."""
    num_envs = engine.get_num_envs()
    com = wp.to_torch(engine._sim_model.body_com).view(num_envs, -1, 3)
    assert char_id == 0, "the root body of another object is not at index 0"
    return com[0, 0].clone()


def mocap_velocities_to_newton(kin_char_model, root_com, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel):
    """Mocap velocities (MimicKit conventions) -> Newton's conventions. Returns (root_vel, dof_vel).

    Root: MimicKit gives the velocity of the root origin, Newton the velocity of the root
    center of mass: v_com = v_origin + ω × (R · root_com).
    Joints: unchanged. Newton's solver integrates and reads back spherical joint velocities
    in the child frame (q0⁻¹·q1), like the mocap (measured in validation/check_heading_env.py, test 2).
    Only Newton's eval_fk reads them in the parent frame (validation/check_disc_features.py, test 1b), and
    it is not used to simulate. Hinge joints and the root angular velocity are the same in both.
    `kin_char_model` and `joint_rot` are kept in the signature for the callers; unused."""
    r = torch_util.quat_rotate(root_rot, root_com.to(root_rot).expand(root_rot.shape[:-1] + (3,)))
    root_vel = root_vel + torch.cross(root_ang_vel, r, dim=-1)
    return root_vel, dof_vel


class ExpertTransitions:
    """Samples expert transitions from the mocap and returns their Φ pairs."""

    def __init__(self, motion_lib, features, root_com, dt=CONTROL_DT):
        self.motion_lib = motion_lib
        self.features = features                 # AMPFeatures
        self.kin = motion_lib._kin_char_model
        self.root_com = root_com
        self.dt = dt

    def state(self, motion_ids, times):
        """Mocap state at these instants, velocities in Newton's conventions -> EtatPerso."""
        rp, rr, rv, ra, jr, dv = self.motion_lib.calc_motion_frame(motion_ids, times)[:6]
        rv, dv = mocap_velocities_to_newton(self.kin, self.root_com, rr, rv, ra, jr, dv)
        body_pos, _ = self.kin.forward_kinematics(rp, rr, jr)
        return EtatPerso(rp, rr, rv, ra, jr, dv, self.features.extraire_key_pos(body_pos))

    def sample(self, n, return_info=False):
        """n transitions -> (n, 2 * dim) = [Φ(s), Φ(s')].
        Clip drawn with the dataset weights, then t uniform in [0, length - dt): s at t, s' at t + dt,
        always inside the same clip. With return_info, also returns (motion_ids, t)."""
        ids = self.motion_lib.sample_motions(n)
        t = self.motion_lib.sample_time(ids, truncate_time=self.dt)
        phi = self.features.phi_paire(self.state(ids, t), self.state(ids, t + self.dt))
        return (phi, ids, t) if return_info else phi
