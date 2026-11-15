"""Benchmark: how many policy steps per second can we simulate on this machine?

Plain Python + MuJoCo (C++), no JAX. N humanoids are stepped in parallel by
MuJoCo's multithreaded `rollout` module (the threads run in C++). Zero control,
flat floor. 1 policy step = sim_hz / 30 physics steps.

Optionally (if PyTorch is installed) also times the forward pass of a
1024-512 MLP policy on the same batch, to see whether the network or the
physics will be the bottleneck.

Usage:
    python scripts/sim_benchmark.py                          # threads 1 2 4 8, 1024 envs
    python scripts/sim_benchmark.py --threads 1 16 32 --envs 4096
"""

import argparse
import os
import time

import mujoco
import numpy as np
from mujoco import rollout

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_XML = os.path.join(ROOT, "data", "assets", "humanoid", "humanoid.xml")
POLICY_HZ = 30


def build_model(xml_path, sim_hz):
    # humanoid.xml has no floor: wrap it in a small scene, as in the notebook
    scene = f"""
    <mujoco>
      <include file="{os.path.abspath(xml_path)}"/>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 .05"/>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(scene)
    model.opt.timestep = 1.0 / sim_hz
    return model


def initial_states(model, n, rng):
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    data = mujoco.MjData(model)
    states = np.empty((n, mujoco.mj_stateSize(model, spec)))
    for i in range(n):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.9
        data.qpos[7:] += 0.1 * rng.standard_normal(model.nq - 7)
        mujoco.mj_forward(model, data)
        mujoco.mj_getState(model, data, states[i], spec)
    return states


def bench_physics(model, states, n_threads, sim_hz, n_policy_steps):
    substeps = sim_hz // POLICY_HZ
    datas = [mujoco.MjData(model) for _ in range(n_threads)]
    ctrl = np.zeros((len(states), substeps, model.nu))
    state = states.copy()
    rollout.rollout(model, datas, state, ctrl, nstep=substeps, persistent_pool=True)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_policy_steps):
        traj, _ = rollout.rollout(model, datas, state, ctrl, nstep=substeps, persistent_pool=True)
        state = traj[:, -1, :]
    dt = time.perf_counter() - t0
    rollout.shutdown_persistent_pool()
    nan_frac = float(np.isnan(state).any(axis=1).mean())
    return len(states) * n_policy_steps / dt, nan_frac


def bench_policy(n_envs, obs_dim=200, act_dim=28, n_iters=50):
    try:
        import torch
    except ImportError:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 1024), torch.nn.ReLU(),
        torch.nn.Linear(1024, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, act_dim),
    ).to(device)
    obs = np.random.randn(n_envs, obs_dim).astype(np.float32)
    with torch.no_grad():
        net(torch.from_numpy(obs).to(device))  # warm-up
        t0 = time.perf_counter()
        for _ in range(n_iters):
            # include the CPU <-> device copies, as in the real loop
            act = net(torch.from_numpy(obs).to(device)).cpu().numpy()
        dt = time.perf_counter() - t0
    return device, n_envs * n_iters / dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sim-hz", type=int, nargs="+", default=[120])
    parser.add_argument("--steps", type=int, default=30, help="policy steps per measurement")
    args = parser.parse_args()

    print(f"mujoco {mujoco.__version__} | {os.cpu_count()} CPUs visible "
          f"(SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', '-')})")
    rng = np.random.default_rng(0)

    for sim_hz in args.sim_hz:
        assert sim_hz % POLICY_HZ == 0, "sim_hz must be a multiple of 30"
        model = build_model(args.xml, sim_hz)
        states = initial_states(model, args.envs, rng)
        print(f"\nphysics {sim_hz} Hz ({sim_hz // POLICY_HZ} substeps per policy step), {args.envs} envs")
        print(f"{'threads':>8} {'policy steps/s':>16} {'100M samples':>13} {'600M samples':>13} {'NaN':>5}")
        for n in args.threads:
            sps, nan = bench_physics(model, states, n, sim_hz, args.steps)
            print(f"{n:8d} {sps:16,.0f} {1e8 / sps / 3600:12.1f}h {6e8 / sps / 3600:12.1f}h {nan:5.0%}")

    pol = bench_policy(args.envs)
    if pol is None:
        print("\nPyTorch not installed: policy forward pass not measured.")
    else:
        device, sps = pol
        print(f"\npolicy MLP 1024-512 forward on {device}: {sps:,.0f} samples/s (batch {args.envs})")
    print("\nNote: physics (+ policy forward) only. PPO and discriminator updates add to the training time.")


if __name__ == "__main__":
    main()
