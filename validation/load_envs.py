"""Environments shared by the validation scripts, built once per Python process.

This module only defines and builds the environments (which config files, how many copies);
the tests themselves are in the validation/check_*.py scripts.

    steering_env()       MimicKit's AMP steering env, unchanged      phase 1
    heading_env()        our Target Heading env (amp/heading_env.py)  phase 2
    heading_env_nopd()   the same, without PD, 1/240 s per step       phase 2, test 2

Each getter builds its environment the first time it is called, then returns the same object:
running several tests in one process (a script with --tests 1 2, a notebook, `python -i`)
builds each environment only once. Every test resets the environment it uses.

    import load_envs                       # sets up the paths (MimicKit, project) and the cwd
    env = load_envs.heading_env()          # built here
    env = load_envs.heading_env()          # same object, not rebuilt

Importing this module changes the working directory to MimicKit's root, which MimicKit needs
(it reads its files with relative paths).
"""

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MIMICKIT = PROJECT / "reference" / "MimicKit"
os.chdir(MIMICKIT)
for p in (str(PROJECT), str(MIMICKIT / "mimickit")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / "reference" / "warpcache"))

import envs.env_builder as env_builder        # MimicKit: imported first, it sets the simulators' import order

from amp.heading_env import build_heading_env

DEVICE = "cuda:0"
NOPD_ENGINE_FILE = str(PROJECT / "configs" / "newton_test_nopd_240hz.yaml")

_cache = {}


def _get(key, build):
    if key not in _cache:
        _cache[key] = build()
    return _cache[key]


def steering_env(num_envs=64):
    """MimicKit's AMP steering environment (TaskSteeringEnv), unchanged. Used by phase 1, which
    only takes its Newton engine, skeleton and clips (validation/check_disc_features.py, build_env).

    env_builder.build_env arguments:
      - env file: skeleton, key bodies, episode length, initial pose, reward weights, and the
        motion file (list of clips with their sampling weight);
      - engine file: newton, PD targets (control_mode: pos), policy at 30 Hz, physics at 240 Hz;
      - num_envs: number of parallel copies of the world (phase 1: one tested state per copy);
      - visualize: no 3D viewer."""
    return _get(("steering", num_envs),
                lambda: env_builder.build_env("data/envs/amp_steering_humanoid_env.yaml",
                                              "data/engines/newton_engine.yaml", num_envs, DEVICE, visualize=False))


def heading_env(num_envs=64, env_file=None):
    """Target Heading environment (amp/heading_env.py), PD control, policy at 30 Hz.
    Used by all phase 2 tests. env_file: another env config (e.g. the env_config.yaml of a run,
    whose dataset can differ); default configs/run1/locomotion_env.yaml."""
    env_file = str(env_file) if env_file else None
    return _get(("heading", num_envs, env_file), lambda: build_heading_env(num_envs, DEVICE, env_file=env_file))


def heading_env_nopd(num_envs=16):
    """Same environment without PD control, one 1/240 s physics step per env step
    (configs/newton_test_nopd_240hz.yaml). Used by phase 2 test 2 to read the solver's
    velocity conventions."""
    return _get(("heading_nopd", num_envs),
                lambda: build_heading_env(num_envs, DEVICE, engine_file=NOPD_ENGINE_FILE))
