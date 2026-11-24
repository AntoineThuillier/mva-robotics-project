"""Target Heading environment (phase 2, CADRAGE.md; article section 7, eq. 11).

    env = build_heading_env(num_envs=4096, device="cuda:0")
    obs, info = env.reset()
    obs, reward, done, info = env.step(actions)      # actions = PD targets, 30 Hz

Inherits MimicKit's TaskSteeringEnv (physics, PD, character, fall detection, reference
state init from the mocap) and only changes what differs from the article:

    goal          g = (target direction in the character's heading frame, v*), 3 numbers;
                  MimicKit also adds a "facing" direction (removed)
    schedule      DeepMimic's (SceneHeadingAMP.cpp): every 0.2-0.5 s the heading drifts by
                  N(0, 0.15 rad), with prob. 0.01 a sharp turn U(-pi, pi), with prob. 0.02 a
                  new v* U(1, 5) m/s. MimicKit: new random direction every 4-7 s
    reward        eq. 11, DeepMimic version: r_G = exp(-0.25 (v* - d*.v_com)^2), v_com = mean
                  velocity of the center of mass over the policy step, r_G = 0 when moving
                  backwards. MimicKit: root velocity, 2D vector error, + a facing term
    reset         root velocity converted from the root origin (mocap) to the root center of
                  mass (Newton): v_com = v_origin + w x (R r_com). MimicKit writes it unconverted.
                  Joint velocities are NOT converted: Newton's MuJoCo solver integrates
                  spherical joint velocities in the child frame, like the mocap
                  (validation/check_heading_env.py, test 2b; same convention as amp/expert.py)
    episode       20 s (configs/run1/locomotion_env.yaml)
    mocap height  optional (align_mocap_feet, off by default): each clip shifted vertically so that
                  its stance feet are at the simulated contact level (amp/foot_align.py)
    disc obs      phase 4: info["disc_obs"] = our phi_paire(s, s') (amp/disc_features.py), demos
                  from amp/expert.py, instead of MimicKit's features (10 states in its config)

Goal observation: DeepMimic writes (cos D, -sin D, v*) with D = target heading - character
heading, in its y-up frame where a heading h points to (cos h, 0, -sin h). In MimicKit's z-up
frame the same quantity, the target direction seen from the character, is (cos D, sin D).

Like the rest of amp/, import with MimicKit's `mimickit/` folder in sys.path, and run from
MimicKit's root (its config files use relative paths).
"""

from pathlib import Path

import numpy as np
import torch
import warp as wp

import envs.env_builder as env_builder        # imported first: sets the simulators' import order
import envs.smp_env as smp_env
import envs.task_steering_env as task_steering_env
import util.torch_util as torch_util

from amp.disc_features import AMPFeatures, EtatPerso
from amp.expert import ExpertTransitions, root_com_from_engine
from amp.foot_align import align_clips

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = "configs/run1/locomotion_env.yaml"     # relative to the project root
DEFAULT_ENGINE_FILE = "data/engines/newton_engine.yaml"         # relative to MimicKit's root


def build_heading_env(num_envs, device, env_file=None, engine_file=DEFAULT_ENGINE_FILE,
                      visualize=False, record_video=False):
    """MimicKit's env_builder only knows MimicKit's envs: build ours directly."""
    if env_file is None:
        env_file = str(PROJECT / DEFAULT_ENV_FILE)
    env_config, engine_config = env_builder.load_configs(env_file, engine_file)
    return TaskHeadingEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs,
                          device=device, visualize=visualize, record_video=record_video)


class TaskHeadingEnv(task_steering_env.TaskSteeringEnv):

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):

        self._heading_drift_std = float(env_config["heading_drift_std"])
        self._sharp_turn_prob = float(env_config["sharp_turn_prob"])
        self._speed_change_prob = float(env_config["speed_change_prob"])
        self._reward_vel_scale = float(env_config["reward_vel_scale"])
        self.convert_init_vel = env_config.get("convert_init_vel", True)   # False only for tests
        # mocap clips shifted vertically to the simulated contact level (amp/foot_align.py), off by default
        self._align_mocap_feet = bool(env_config.get("align_mocap_feet", False))
        self._foot_contact_height = float(env_config.get("foot_contact_height", 0.0))
        self.clip_height_offsets = None        # (num_motions,) m, set when align_mocap_feet

        # keys TaskSteeringEnv reads but that we do not use
        env_config = dict(env_config)
        env_config.setdefault("reward_steering_tar_w", 1.0)
        env_config.setdefault("reward_steering_face_w", 0.0)
        env_config.setdefault("reward_steering_vel_scale", self._reward_vel_scale)

        # Inherit from class "steering" from MimicKit
        super().__init__(env_config=env_config, engine_config=engine_config, num_envs=num_envs,
                         device=device, visualize=visualize, record_video=record_video)
        sort_contact_sensor_pairs(self._engine._ground_contact_sensor)
        return

    # ---------- mocap: vertical alignment of the clips (amp/foot_align.py) ----------
    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)
        if self._align_mocap_feet:
            self.clip_height_offsets = align_clips(self._motion_lib, self._kin_char_model, self._foot_contact_height)
            o = self.clip_height_offsets * 100
            print(f"align_mocap_feet: {len(o)} clips shifted to a stance level of {100 * self._foot_contact_height:+.1f} cm, "
                  f"offsets {o.min():+.1f} to {o.max():+.1f} cm (median {o.median():+.1f})")
        return

    # ---------- buffers ----------
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        num_envs = self.get_num_envs()
        self._tar_heading = torch.zeros(num_envs, device=self._device)
        self._prev_com = torch.zeros(num_envs, 3, device=self._device)
        # goal in force during the current step (the goal may be updated before the reward is computed)
        self._step_tar_dir = torch.zeros(num_envs, 2, device=self._device)
        self._step_tar_speed = torch.zeros(num_envs, device=self._device)

        # mass and center of mass of each body of the character, as Newton uses them
        char_id = self._get_char_id()
        assert char_id == 0, "the character's bodies are assumed to come first in each world"
        nb = self._engine.get_obj_num_bodies(char_id)
        model = self._engine.get_sim_model()
        self._body_mass = wp.to_torch(model.body_mass).view(num_envs, -1)[0, :nb].clone()
        self._body_com = wp.to_torch(model.body_com).view(num_envs, -1, 3)[0, :nb].clone()
        return

    def calc_com(self):
        """Center of mass of the character, world frame, (num_envs, 3)."""
        char_id = self._get_char_id()
        return calc_com(self._engine.get_body_pos(char_id), self._engine.get_body_rot(char_id),
                        self._body_com, self._body_mass)

    # ---------- reset: mocap root velocity in Newton's convention ----------
    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        if self.convert_init_vel:
            root_vel = root_origin_vel_to_com_vel(self._body_com[0], self._ref_root_rot[env_ids],
                                                  self._ref_root_vel[env_ids], self._ref_root_ang_vel[env_ids])
            self._engine.set_root_vel(env_ids, self._get_char_id(), root_vel)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)          # character (RSI), then goal (_reset_task)
        if len(env_ids) > 0:
            # COM from the reset pose, computed with the kinematic model (Newton's body
            # positions are only refreshed at the next simulation step)
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                self._ref_root_pos[env_ids], self._ref_root_rot[env_ids], self._ref_joint_rot[env_ids])
            self._prev_com[env_ids] = calc_com(body_pos, body_rot, self._body_com, self._body_mass)
        return

    # ---------- discriminator observations: our Φ (phase 4, step 2) ----------
    # MimicKit's AMPEnv keeps the last num_disc_obs_steps (= 2) states of the character, read
    # from Newton after each policy step, and gives info["disc_obs"] to the agent; we only replace
    # what is computed from them: phi_paire(s, s') (amp/disc_features.py) instead of MimicKit's
    # features, and expert pairs from amp/expert.py.

    def _amp_features(self):
        """AMPFeatures and ExpertTransitions, built on first use: MimicKit asks for the shape of
        the discriminator observations while the env is still being built."""
        if not hasattr(self, "_phi"):
            self._phi = AMPFeatures(self._kin_char_model)
            self._expert = ExpertTransitions(self._motion_lib, self._phi, root_com_from_engine(self._engine))
        return self._phi, self._expert

    def fetch_disc_obs_demo(self, num_samples):
        """Expert pairs [Φ(s), Φ(s')] from the mocap, velocities in Newton's convention. (n, 210)."""
        _, expert = self._amp_features()
        return expert.sample(num_samples)

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
        return

    def _reset_disc_hist(self, env_ids):
        """MimicKit fills the history with the mocap frames at t - 1/30 s and t: convert their root
        velocity to Newton's convention, like the reset state (_ref_state_init)."""
        super()._reset_disc_hist(env_ids)
        if self.convert_init_vel and len(env_ids) > 0:
            rr = self._disc_hist_root_rot.get_all()[env_ids]
            rv = self._disc_hist_root_vel.get_all()[env_ids]
            ra = self._disc_hist_root_ang_vel.get_all()[env_ids]
            self._disc_hist_root_vel.fill(env_ids, root_origin_vel_to_com_vel(self._body_com[0], rr, rv, ra))
        return

    # ---------- goal ----------
    def _reset_tar(self, env_ids):
        n = env_ids.shape[0]
        self._tar_heading[env_ids] = (2 * torch.rand(n, device=self._device) - 1) * np.pi
        self._tar_speed[env_ids] = self._rand_speed(n)
        self._sync_tar_dir(env_ids)
        return

    # _reset_task: MimicKit's (_reset_tar, then next update in U(tar_change_time_min, max))

    def _update_task(self):
        env_ids = (self._time_buf >= self._tar_change_times).nonzero(as_tuple=False).flatten()
        n = env_ids.shape[0]
        if n > 0:
            sharp = torch.rand(n, device=self._device) < self._sharp_turn_prob
            delta = torch.where(sharp,
                                (2 * torch.rand(n, device=self._device) - 1) * np.pi,
                                self._heading_drift_std * torch.randn(n, device=self._device))
            self._tar_heading[env_ids] = torch_util.normalize_angle(self._tar_heading[env_ids] + delta)

            change = torch.rand(n, device=self._device) < self._speed_change_prob
            self._tar_speed[env_ids] = torch.where(change, self._rand_speed(n), self._tar_speed[env_ids])

            self._sync_tar_dir(env_ids)
            self._tar_change_times[env_ids] = self._time_buf[env_ids] + self._rand_update_delay(n)

        if self._visualize:
            self._update_marker(self._env_ids)
        return

    def _sync_tar_dir(self, env_ids):
        h = self._tar_heading[env_ids]
        self._tar_dir[env_ids] = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
        self._face_dir[env_ids] = self._tar_dir[env_ids]      # only used by MimicKit's markers
        return

    def _rand_speed(self, n):
        return self._tar_speed_min + (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self._device)

    def _rand_update_delay(self, n):
        return self._tar_change_time_min + (self._tar_change_time_max - self._tar_change_time_min) * torch.rand(n, device=self._device)

    # ---------- observation: character state (MimicKit) + g ----------
    def _compute_obs(self, env_ids=None):
        obs = smp_env.SMPEnv._compute_obs(self, env_ids)     # skips TaskSteeringEnv's goal obs
        root_rot = self._engine.get_root_rot(self._get_char_id())
        tar_dir, tar_speed = self._tar_dir, self._tar_speed
        if env_ids is not None:
            root_rot, tar_dir, tar_speed = root_rot[env_ids], tar_dir[env_ids], tar_speed[env_ids]
        return torch.cat([obs, compute_heading_goal_obs(root_rot, tar_dir, tar_speed)], dim=-1)

    # ---------- reward ----------
    def _pre_physics_step(self, actions):
        smp_env.SMPEnv._pre_physics_step(self, actions)       # skips TaskSteeringEnv's root recording
        self._step_tar_dir[:] = self._tar_dir
        self._step_tar_speed[:] = self._tar_speed
        return

    def _update_reward(self):
        com = self.calc_com()
        self._reward_buf[:] = compute_heading_reward(self._prev_com, com, self._step_tar_dir,
                                                     self._step_tar_speed, self._engine.get_timestep(),
                                                     self._reward_vel_scale)
        self._prev_com[:] = com
        return


# ===================== fix of Newton's contact sensor =====================

def sort_contact_sensor_pairs(sensor):
    """Fixes fall detection (Newton 1.0.0 bug, TODO-antoine.md). For each contact, the sensor finds
    its (shape, shape) pair by binary search in a list that must be sorted, but builds that list by
    concatenating the sorted list of each world. The ground is shape 0, shared by all worlds, so
    the concatenation is not sorted: some pairs are never found and their ground forces are lost
    (whole characters for most num_envs that are not powers of 2). Sorts the list globally, with
    the readings of each pair. Call once, after the env is built."""
    sp = sensor._sp_sorted.numpy()
    order = np.lexsort((sp[:, 1], sp[:, 0]))
    reading = sensor._sp_reading.numpy()
    offset, count = sensor._sp_ep_offset.numpy(), sensor._sp_ep_count.numpy()
    new_count = count[order].astype(np.int32)
    new_reading = np.concatenate([reading[offset[i]:offset[i] + count[i]] for i in order])
    new_offset = np.concatenate([[0], np.cumsum(new_count)[:-1]]).astype(np.int32)
    sensor._sp_sorted = wp.array(sp[order], dtype=wp.vec2i, device=sensor.device)
    sensor._sp_reading = wp.array(new_reading, dtype=wp.vec2i, device=sensor.device)
    sensor._sp_ep_offset = wp.array(new_offset, dtype=wp.int32, device=sensor.device)
    sensor._sp_ep_count = wp.array(new_count, dtype=wp.int32, device=sensor.device)


# ===================== pure functions (tested in validation/check_heading_env.py) =====================

def root_origin_vel_to_com_vel(root_com, root_rot, root_vel, root_ang_vel):
    """Velocity of the root origin (mocap) -> velocity of the root center of mass (Newton):
    v_com = v_origin + w x (R r_com). root_com (3,) in the root frame."""
    r = torch_util.quat_rotate(root_rot, root_com.to(root_rot).expand(root_rot.shape[:-1] + (3,)))
    return root_vel + torch.cross(root_ang_vel, r, dim=-1)


def calc_com(body_pos, body_rot, body_com, body_mass):
    """body_pos (..., B, 3), body_rot (..., B, 4) xyzw: origin and orientation of each body;
    body_com (B, 3): center of mass of each body in its own frame; body_mass (B,).
    Returns the center of mass of the whole character (..., 3)."""
    com_local = body_com.to(body_pos).expand(body_pos.shape)
    com_world = body_pos + torch_util.quat_rotate(body_rot, com_local)
    w = body_mass.to(body_pos)
    return (com_world * w.unsqueeze(-1)).sum(-2) / w.sum()


def compute_heading_goal_obs(root_rot, tar_dir, tar_speed):
    """g = (target direction in the character's heading frame (2), v*). Returns (..., 3)."""
    tar_dir3 = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., :1])], dim=-1)
    local = torch_util.quat_rotate(torch_util.calc_heading_quat_inv(root_rot), tar_dir3)
    return torch.cat([local[..., :2], tar_speed.unsqueeze(-1)], dim=-1)


def compute_heading_reward(prev_com, com, tar_dir, tar_speed, dt, vel_scale):
    """Eq. 11, DeepMimic version: mean COM velocity over the step, projected on the target
    direction; 0 when moving backwards. Returns (...,)."""
    v = (com[..., :2] - prev_com[..., :2]) / dt
    v_proj = (tar_dir * v).sum(-1)
    r = torch.exp(-vel_scale * (tar_speed - v_proj) ** 2)
    return torch.where(v_proj > 0, r, torch.zeros_like(r))
