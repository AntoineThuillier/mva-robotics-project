"""Target Location environment (article section 7 and eq. 12; DeepMimic's SceneTargetAMP.cpp).

    env = build_location_env(num_envs=4096, device="cuda:0", env_file="configs/location/locomotion_env.yaml")

The character must reach a target position x* on the ground. Inherits MimicKit's TaskLocationEnv
(target marker, target sampling) and our AMP additions (amp/amp_common.py: Φ for D, root velocity
converted at reset, COM, aligned mocap feet, contact sensor fix), and follows DeepMimic's code
(args/train_amp_target_humanoid3d_locomotion_args.txt and SceneTargetAMP.cpp), which produced the
article's numbers:

    schedule   new target every U(5, 10) s, at distance U(0, 10) m and angle U(0, 2 pi) from the
               root (MimicKit: same; DeepMimic: rand_target_time 5-10, max_target_dist 10)
    goal       g = (unit direction to the target in the character's heading frame (2), distance (1));
               MimicKit uses the 2D relative position instead
    reward     r_G = w_pos exp(-k_pos ||x* - x_root||^2) + w_vel r_vel, horizontal distances,
               r_vel = 1 within success_dist of the target,
                     = 0 when the COM moves away from the target,
                     = exp(-k_vel max(0, v* - d*.v_com)^2) otherwise,
               v_com = mean COM velocity over the policy step, d* = horizontal unit vector from the
               COM to the target, v* a minimum speed (moving faster is not penalized);
               r_G = 0 at a fall or beyond fail_dist.
               DeepMimic's code: w_pos 0.6, w_vel 0.4, k_pos 0.5, k_vel = 4 / v*^2, v* = 1 m/s.
               The article's eq. 12 writes w_pos 0.7, w_vel 0.3, k_vel 1: set these in the config
               (loc_vel_err_scale is the numerator: k_vel = loc_vel_err_scale / v*^2).
    v*         fixed (tar_speed), or, when tar_speed_dist_gain > 0, growing with the distance to the
               target at the start of the step: v* = clamp(gain * d, tar_speed, tar_speed_max)
               (our addition: with v* = 1 m/s, running brings no reward, and the Run and Locomotion
               policies never really run; configs/location/README.md).
               MimicKit: 0.5 / 0.4 + 0.1 facing term, root velocity.
    end        fall, 20 s, or the root more than fail_dist = 15 m from the target (failure)

The target and v* in force during a policy step are used for its reward, even if a new target is
drawn just after the step (as for Target Heading).
"""

from pathlib import Path

import torch

import envs.env_builder as env_builder        # imported first: sets the simulators' import order
import envs.base_env as base_env
import envs.smp_env as smp_env
import envs.task_location_env as task_location_env
import util.torch_util as torch_util

from amp.amp_common import AMPEnvCommon

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = "configs/location/locomotion_env.yaml"       # relative to the project root
DEFAULT_ENGINE_FILE = "data/engines/newton_engine.yaml"          # relative to MimicKit's root


def build_location_env(num_envs, device, env_file=None, engine_file=DEFAULT_ENGINE_FILE,
                       visualize=False, record_video=False):
    if env_file is None:
        env_file = str(PROJECT / DEFAULT_ENV_FILE)
    env_config, engine_config = env_builder.load_configs(env_file, engine_file)
    return TaskLocationEnv(env_config=env_config, engine_config=engine_config, num_envs=num_envs,
                           device=device, visualize=visualize, record_video=record_video)


class TaskLocationEnv(AMPEnvCommon, task_location_env.TaskLocationEnv):

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._init_common(env_config)
        self._loc_pos_w = float(env_config.get("loc_pos_reward_w", 0.6))
        self._loc_vel_w = float(env_config.get("loc_vel_reward_w", 0.4))
        self._loc_pos_scale = float(env_config.get("loc_pos_err_scale", 0.5))
        self._loc_vel_scale = float(env_config.get("loc_vel_err_scale", 4.0))
        self._success_dist = float(env_config.get("success_dist", 0.5))
        self._fail_dist = float(env_config.get("fail_dist", 15.0))
        self._tar_speed_gain = float(env_config.get("tar_speed_dist_gain", 0.0))
        self._tar_speed_max = float(env_config.get("tar_speed_max", env_config["tar_speed"]))
        super().__init__(env_config=env_config, engine_config=engine_config, num_envs=num_envs,
                         device=device, visualize=visualize, record_video=record_video)
        self._finish_common()

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self._step_tar_pos = torch.zeros(self.get_num_envs(), 3, device=self._device)
        self._step_tar_speed = torch.zeros(self.get_num_envs(), device=self._device)

    # ---------- observation: character state (MimicKit) + g ----------
    def _compute_obs(self, env_ids=None):
        obs = smp_env.SMPEnv._compute_obs(self, env_ids)     # skips MimicKit's 2D target observation
        char_id = self._get_char_id()
        root_pos, root_rot, tar_pos = self._engine.get_root_pos(char_id), self._engine.get_root_rot(char_id), self._tar_pos
        if env_ids is not None:
            root_pos, root_rot, tar_pos = root_pos[env_ids], root_rot[env_ids], tar_pos[env_ids]
        return torch.cat([obs, compute_location_goal_obs(root_pos, root_rot, tar_pos)], dim=-1)

    # ---------- reward and end of episode ----------
    def _pre_physics_step(self, actions):
        smp_env.SMPEnv._pre_physics_step(self, actions)       # skips MimicKit's root recording
        self._step_tar_pos[:] = self._tar_pos
        dist = horizontal_dist(self._tar_pos, self._engine.get_root_pos(self._get_char_id()))
        self._step_tar_speed[:] = compute_tar_speed(dist, self._tar_speed, self._tar_speed_max, self._tar_speed_gain)

    def _update_reward(self):
        com = self.calc_com()
        root_pos = self._engine.get_root_pos(self._get_char_id())
        self._reward_buf[:] = compute_location_reward(
            root_pos, self._prev_com, com, self._step_tar_pos, self._step_tar_speed, self._engine.get_timestep(),
            self._loc_pos_w, self._loc_vel_w, self._loc_pos_scale, self._loc_vel_scale, self._success_dist)
        self._prev_com[:] = com

    def _update_done(self):
        super()._update_done()                                # falls and the 20 s limit
        root_pos = self._engine.get_root_pos(self._get_char_id())
        far = horizontal_dist(self._step_tar_pos, root_pos) > self._fail_dist
        self._done_buf[far & (self._done_buf == base_env.DoneFlags.NULL.value)] = base_env.DoneFlags.FAIL.value
        # DeepMimic: no task reward once fallen or too far
        self._reward_buf[(self._done_buf == base_env.DoneFlags.FAIL.value)] = 0.0


# ===================== pure functions (tested in validation/check_location_env.py) =====================

def horizontal_dist(a, b):
    return torch.linalg.norm((a - b)[..., :2], dim=-1)


def compute_tar_speed(dist, v_min, v_max, gain):
    """v* = clamp(gain * dist, v_min, v_max); v_min when gain = 0. (...,)."""
    if gain <= 0:
        return torch.full_like(dist, v_min)
    return torch.clamp(gain * dist, min=v_min, max=v_max)


def compute_location_goal_obs(root_pos, root_rot, tar_pos):
    """g = (unit direction to the target in the heading frame (2), horizontal distance). (..., 3)."""
    delta = tar_pos - root_pos
    delta = torch.cat([delta[..., :2], torch.zeros_like(delta[..., :1])], dim=-1)
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    local = torch_util.quat_rotate(torch_util.calc_heading_quat_inv(root_rot), delta)[..., :2]
    unit = torch.where(dist > 1e-4, local / dist.clamp_min(1e-4), torch.tensor([1.0, 0.0], device=local.device))
    return torch.cat([unit, dist], dim=-1)


def compute_location_reward(root_pos, prev_com, com, tar_pos, tar_speed, dt,
                            pos_w, vel_w, pos_scale, vel_scale, success_dist):
    """r_G of DeepMimic's SceneTargetAMP.cpp (falls and fail distance are handled in _update_done). (...,).
    tar_speed: v*, a float or a tensor (...,); vel_scale is divided by v*^2 (DeepMimic: 4 / v*^2)."""
    pos_reward = torch.exp(-pos_scale * horizontal_dist(tar_pos, root_pos) ** 2)

    to_tar = (tar_pos - com)[..., :2]
    d = to_tar / torch.linalg.norm(to_tar, dim=-1, keepdim=True).clamp_min(1e-4)
    v = torch.sum(d * (com - prev_com)[..., :2], dim=-1) / dt          # mean COM speed towards the target
    vel_reward = torch.exp(-vel_scale / tar_speed ** 2 * torch.clamp_min(tar_speed - v, 0.0) ** 2)
    vel_reward = torch.where(v < 0, torch.zeros_like(vel_reward), vel_reward)
    near = horizontal_dist(tar_pos, root_pos) < success_dist
    vel_reward = torch.where(near, torch.ones_like(vel_reward), vel_reward)
    return pos_w * pos_reward + vel_w * vel_reward
