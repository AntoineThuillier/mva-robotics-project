"""What our AMP environments add to MimicKit's, shared by the tasks (mixin).

    class TaskLocationEnv(AMPEnvCommon, task_location_env.TaskLocationEnv): ...

Put AMPEnvCommon first in the bases: its methods call super(), i.e. MimicKit's env. It brings:

    mocap height  optional (align_mocap_feet): each clip shifted vertically so that its stance
                  feet are at the simulated contact level (amp/foot_align.py)
    reset         root velocity converted from the root origin (mocap) to the root center of
                  mass (Newton); joint velocities unchanged (same convention, see heading_env.py)
    COM           mass and center of mass of each body, calc_com(), and the COM at the start of
                  each policy step (self._prev_com), for task rewards on the mean COM velocity
    disc obs      info["disc_obs"] = our phi_paire(s, s') (amp/disc_features.py), demos from
                  amp/expert.py, instead of MimicKit's features
    contacts      fix of Newton's contact sensor (fall detection), heading_env.sort_contact_sensor_pairs

Same code as amp/heading_env.py (TaskHeadingEnv), which keeps its own copy for now so that the
heading runs in progress are not touched.
"""

import torch
import warp as wp

from amp.disc_features import AMPFeatures, EtatPerso
from amp.expert import ExpertTransitions, root_com_from_engine
from amp.foot_align import align_clips
from amp.heading_env import calc_com, root_origin_vel_to_com_vel, sort_contact_sensor_pairs


class AMPEnvCommon:

    def _init_common(self, env_config):
        """Call in __init__ before MimicKit's __init__ (it reads these while building)."""
        self.convert_init_vel = env_config.get("convert_init_vel", True)   # False only for tests
        self._align_mocap_feet = bool(env_config.get("align_mocap_feet", False))
        self._foot_contact_height = float(env_config.get("foot_contact_height", 0.0))
        self.clip_height_offsets = None

    def _finish_common(self):
        """Call at the end of __init__, after MimicKit's __init__."""
        sort_contact_sensor_pairs(self._engine._ground_contact_sensor)

    # ---------- mocap: vertical alignment of the clips ----------
    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)
        if self._align_mocap_feet:
            self.clip_height_offsets = align_clips(self._motion_lib, self._kin_char_model, self._foot_contact_height)
            o = self.clip_height_offsets * 100
            print(f"align_mocap_feet: {len(o)} clips shifted to a stance level of {100 * self._foot_contact_height:+.1f} cm, "
                  f"offsets {o.min():+.1f} to {o.max():+.1f} cm (median {o.median():+.1f})")

    # ---------- buffers: mass and COM of the bodies ----------
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        num_envs = self.get_num_envs()
        self._prev_com = torch.zeros(num_envs, 3, device=self._device)
        char_id = self._get_char_id()
        assert char_id == 0, "the character's bodies are assumed to come first in each world"
        nb = self._engine.get_obj_num_bodies(char_id)
        model = self._engine.get_sim_model()
        self._body_mass = wp.to_torch(model.body_mass).view(num_envs, -1)[0, :nb].clone()
        self._body_com = wp.to_torch(model.body_com).view(num_envs, -1, 3)[0, :nb].clone()

    def calc_com(self):
        """Center of mass of the character, world frame, (num_envs, 3)."""
        char_id = self._get_char_id()
        return calc_com(self._engine.get_body_pos(char_id), self._engine.get_body_rot(char_id),
                        self._body_com, self._body_mass)

    # ---------- reset ----------
    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        if self.convert_init_vel:
            root_vel = root_origin_vel_to_com_vel(self._body_com[0], self._ref_root_rot[env_ids],
                                                  self._ref_root_vel[env_ids], self._ref_root_ang_vel[env_ids])
            self._engine.set_root_vel(env_ids, self._get_char_id(), root_vel)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            # COM of the reset pose, from the kinematic model (Newton's body positions are only
            # refreshed at the next simulation step)
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                self._ref_root_pos[env_ids], self._ref_root_rot[env_ids], self._ref_joint_rot[env_ids])
            self._prev_com[env_ids] = calc_com(body_pos, body_rot, self._body_com, self._body_mass)

    # ---------- discriminator observations: our Φ ----------
    def _amp_features(self):
        """AMPFeatures and ExpertTransitions, built on first use (MimicKit asks for the shape of the
        discriminator observations while the env is being built)."""
        if not hasattr(self, "_phi"):
            self._phi = AMPFeatures(self._kin_char_model)
            self._expert = ExpertTransitions(self._motion_lib, self._phi, root_com_from_engine(self._engine))
        return self._phi, self._expert

    def fetch_disc_obs_demo(self, num_samples):
        """Expert pairs [Φ(s), Φ(s')] from the mocap, velocities in Newton's convention. (n, 210)."""
        return self._amp_features()[1].sample(num_samples)

    def _disc_hist_state(self, k, env_ids=None):
        """State k of the history (-2 = s, -1 = s') as an EtatPerso."""
        phi, _ = self._amp_features()
        hist = [self._disc_hist_root_pos, self._disc_hist_root_rot, self._disc_hist_root_vel,
                self._disc_hist_root_ang_vel, self._disc_hist_joint_rot, self._disc_hist_dof_vel,
                self._disc_hist_body_pos]
        x = [h.get_all()[:, k] for h in hist]            # get_all: oldest first
        if env_ids is not None:
            x = [v[env_ids] for v in x]
        rp, rr, rv, ra, jr, dv, bp = x
        return EtatPerso(rp, rr, rv, ra, jr, dv, phi.extraire_key_pos(bp))

    def _update_disc_obs(self, env_ids=None):
        phi, _ = self._amp_features()
        disc_obs = phi.phi_paire(self._disc_hist_state(-2, env_ids), self._disc_hist_state(-1, env_ids))
        if env_ids is None:
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

    def _reset_disc_hist(self, env_ids):
        """MimicKit fills the history with the mocap frames at t - 1/30 s and t: convert their root
        velocity to Newton's convention, like the reset state."""
        super()._reset_disc_hist(env_ids)
        if self.convert_init_vel and len(env_ids) > 0:
            rr = self._disc_hist_root_rot.get_all()[env_ids]
            rv = self._disc_hist_root_vel.get_all()[env_ids]
            ra = self._disc_hist_root_ang_vel.get_all()[env_ids]
            self._disc_hist_root_vel.fill(env_ids, root_origin_vel_to_com_vel(self._body_com[0], rr, rv, ra))
