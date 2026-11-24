"""Phase 4 validation (see CADRAGE.md, section 4, phase 4): our AMP (amp/discriminator.py) on the
Target Heading environment (amp/heading_env.py).

Test 1: the discriminator observations of the environment (step 2). The agent side is our
        phi_paire(s, s') computed from the env's history of states, the expert side comes from
        amp/expert.py, and both use the same features and velocity conventions.
Test 2: the discriminator D and the article's equations (step 3): LSGAN loss (eq. 6),
        gradient penalty (eq. 8), style reward (eq. 7), on known values and on our D.
Test 3: the AMP agent (step 4) on one training iteration: rewards, replay buffer, updates.

    python validation/check_amp.py                 # all tests, 64 envs
    python validation/check_amp.py --tests 1

Run from the project root, with the project venv (.venv).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_envs                                      # first: paths, cwd and environment cache
from load_envs import DEVICE, PROJECT

import torch
import yaml

import check_heading_env
from amp import discriminator as disc

AGENT_CONFIG = PROJECT / "configs" / "run1" / "amp_agent.yaml"
W_GP = 10.0          # article, Table 4

TOLERANCE = 1e-3     # float32 (Newton vs MimicKit forward kinematics, see phase 1)


# ======== Test 1: discriminator observations of the environment ========
def test_disc_obs(env=None):
    """Test 1: info["disc_obs"] of the env is our Φ pair, with the expert's conventions.

    (a) right after a reset (reference state init at a random clip and time t), the history
        holds the mocap frames at t - 1/30 s and t: the env's disc_obs must equal the expert
        pair computed by amp/expert.py at the same clip and times (root velocity converted).
    (b) after real simulation steps (PD tracking of the clip), the env's disc_obs must equal
        phi_paire computed by hand from Newton's state read after the previous step (s) and
        after the current one (s'). Envs that reset in between are skipped.
    (c) shapes: (n, 210) for the agent and for the expert demos.

    Criterion: (a) and (b) max error < TOLERANCE, (c) shapes. Returns True if OK.
    """
    env = env or load_envs.heading_env()
    phi, expert = env._amp_features()
    n = env.get_num_envs()
    dt = env._engine.get_timestep()
    print(f"\n=== Test 1: discriminator observations of the env, {n} envs ===")

    # (a) after a reset
    _, info = env.reset()
    ids, t = env._motion_ids.clone(), env._motion_time_offsets.clone()
    agent = info["disc_obs"].clone()
    ref = phi.phi_paire(expert.state(ids, t - dt), expert.state(ids, t))
    err_a = (agent - ref).abs().max().item()
    # the same without converting the root velocity, to show that the conversion is applied
    raw = env._disc_hist_state(-1)
    raw_vel = env._motion_lib.calc_motion_frame(ids, t)[2]
    conv_diff = (raw.root_vel - raw_vel).norm(dim=-1).mean().item()
    ok_a = err_a < TOLERANCE and conv_diff > 1e-3
    print(f"  (a) after reset, env vs expert: max error {err_a:.1e}; "
          f"root velocity changed by the conversion: {conv_diff:.3f} m/s on average  => {'OK' if ok_a else 'FAIL'}")

    # (b) after simulation steps
    char_id = env._get_char_id()
    eng, kin = env._engine, env._kin_char_model

    def newton_state():
        return phi_state(phi, eng.get_root_pos(char_id), eng.get_root_rot(char_id), eng.get_root_vel(char_id),
                         eng.get_root_ang_vel(char_id), kin.dof_to_rot(eng.get_dof_pos(char_id)),
                         eng.get_dof_vel(char_id), eng.get_body_pos(char_id))

    err_b, checked = 0.0, 0
    _, _, done, _ = env.step(check_heading_env.mocap_targets(env))
    alive = done == 0
    s = clone_state(newton_state())
    for _ in range(5):
        _, _, done, info = env.step(check_heading_env.mocap_targets(env))
        s1 = newton_state()
        keep = alive & (done == 0)
        if keep.any():
            by_hand = phi.phi_paire(select(s, keep), select(s1, keep))
            err_b = max(err_b, (info["disc_obs"][keep] - by_hand).abs().max().item())
            checked += int(keep.sum())
        alive = done == 0
        s = clone_state(s1)
    ok_b = err_b < TOLERANCE and checked > 0
    print(f"  (b) after a simulation step, env vs by hand from Newton: max error {err_b:.1e} "
          f"on {checked} transitions  => {'OK' if ok_b else 'FAIL'}")

    # (c) shapes
    demo = env.fetch_disc_obs_demo(8)
    space = tuple(env.get_disc_obs_space().shape)
    ok_c = space == (2 * phi.dim,) and tuple(demo.shape) == (8, 2 * phi.dim) \
        and tuple(info["disc_obs"].shape) == (n, 2 * phi.dim)
    print(f"  (c) disc obs space {space}, agent {tuple(info['disc_obs'].shape)}, demos {tuple(demo.shape)}"
          f"  => {'OK' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


def phi_state(phi, rp, rr, rv, ra, jr, dv, body_pos):
    from amp.disc_features import EtatPerso
    return EtatPerso(rp, rr, rv, ra, jr, dv, phi.extraire_key_pos(body_pos))


def clone_state(e):
    return type(e)(*(v.clone() for v in vars(e).values()))


def select(e, mask):
    return type(e)(*(v[mask] for v in vars(e).values()))


# ======== Test 2: discriminator and equations 6, 7, 8 ========
def test_disc_equations(env=None):
    """Test 2: amp/discriminator.py, on known values and on our D built from the agent config.

    (a) style reward (eq. 7): 1 at D = 1, 0.75 at D = 0, 0 at D = -1 and below, always in [0, 1].
    (b) LSGAN loss (eq. 6): 0 when D outputs +1 on mocap and -1 on agent; 2 when D outputs 0 on both.
    (c) gradient penalty (eq. 8): for a linear D(phi) = phi.w + b it is exactly (w_gp / 2) ||w||^2;
        0 for a constant D; on our D, finite and differentiable with respect to D's weights (all
        but the output bias, on which dD/dphi does not depend).
    (d) our D: input (n, 210) -> output (n,), 1024-512 hidden layers, fed with expert pairs.

    Criterion: all of them (values to 1e-5). Returns True if OK.
    """
    env = env or load_envs.heading_env()
    torch.manual_seed(0)
    print("\n=== Test 2: discriminator and equations 6, 7, 8 ===")
    checks = {}

    d = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0])
    r = disc.style_reward(d)
    expected = torch.tensor([0.0, 0.0, 0.75, 0.9375, 1.0, 0.75, 0.0])
    checks["(a) style reward at D = -3, -1, 0, 0.5, 1, 2, 5"] = (r - expected).abs().max().item() < 1e-5
    print(f"  (a) r_S at D = {d.tolist()}: {[round(x, 4) for x in r.tolist()]}")

    zero = disc.disc_loss_lsgan(torch.ones(10), -torch.ones(10)).item()
    two = disc.disc_loss_lsgan(torch.zeros(10), torch.zeros(10)).item()
    checks["(b) LSGAN loss"] = abs(zero) < 1e-6 and abs(two - 2) < 1e-6
    print(f"  (b) LSGAN loss: perfect D {zero:.1e} (expected 0), D = 0 everywhere {two:.4f} (expected 2)")

    phi = torch.randn(64, 210, device=DEVICE)
    w, b = torch.randn(210, device=DEVICE), 0.3
    gp_lin = disc.gradient_penalty(lambda x: x @ w + b, phi, W_GP).item()
    gp_expected = 0.5 * W_GP * (w ** 2).sum().item()
    gp_const = disc.gradient_penalty(lambda x: 0 * x.sum(-1) + 1.0, phi, W_GP).item()
    checks["(c) gradient penalty, linear and constant D"] = abs(gp_lin - gp_expected) < 1e-4 * gp_expected and abs(gp_const) < 1e-8
    print(f"  (c) gradient penalty: linear D {gp_lin:.4f} (expected {gp_expected:.4f}), constant D {gp_const:.1e}")

    config = yaml.safe_load(open(AGENT_CONFIG))
    model = disc.HeadingAMPModel(config["model"], env).to(DEVICE)
    demo = env.fetch_disc_obs_demo(256)
    out = model.eval_disc(demo)
    gp = disc.gradient_penalty(model.eval_disc, demo, W_GP)
    gp.backward()
    # every weight of D gets a finite gradient, except the output bias: dD/dphi does not depend on it
    grads_ok = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.get_disc_params() if p is not model._disc_out.bias) \
        and model._disc_out.bias.grad is None
    sizes = [m.out_features for m in model._disc_layers if isinstance(m, torch.nn.Linear)] + [model._disc_out.out_features]
    n_params = sum(p.numel() for p in model.get_disc_params())
    checks["(c) gradient penalty on our D"] = bool(torch.isfinite(gp)) and grads_ok
    checks["(d) our D: shapes and layers"] = tuple(out.shape) == (256,) and sizes == [1024, 512, 1]
    print(f"  (c) gradient penalty on our D (untrained): {gp.item():.3f}; finite gradient for every weight of D "
          f"except the output bias, which the penalty does not depend on: {grads_ok}")
    print(f"  (d) our D: input {tuple(demo.shape)} -> output {tuple(out.shape)}, layers {sizes}, {n_params} parameters")

    ok = all(checks.values())
    for name, c in checks.items():
        if not c:
            print(f"  FAIL: {name}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


# ======== Test 3: the AMP agent, one training iteration ========
def test_agent_iteration(env=None):
    """Test 3: HeadingAMPAgent (step 4), on one real training iteration run by hand.

    (a) construction: D has its own optimizer, and the PPO optimizers of pi and V do not touch D;
        D's input normalizer (fitted once on mocap pairs) gives fresh mocap pairs a mean ~0 and
        a std ~1 per feature.
    (b) rewards: after the rollout, the reward used by GAE equals w_G r_G + w_S r_S, with
        r_S = style_reward(D(normalized disc_obs)) recomputed here from the stored agent pairs.
    (c) replay buffer: it received min(disc_replay_insert, rollout size) agent pairs.
    (d) updates: the PPO update changes pi and V; the update of D changes D and only D's
        optimizer steps; the diagnostics (loss, gradient penalty, accuracies) are finite.

    Criterion: all of them. Returns True if OK.
    """
    import learning.base_agent as base_agent
    env = env or load_envs.heading_env()
    config = yaml.safe_load(open(AGENT_CONFIG))
    torch.manual_seed(0)
    agent = disc.build_heading_amp_agent(config, env, DEVICE)
    print(f"\n=== Test 3: the AMP agent, one training iteration ({env.get_num_envs()} envs) ===")
    checks = {}

    # (a) construction
    disc_ids = {id(p) for p in agent._model.get_disc_params()}
    ppo_ids = {id(p) for p in agent._model.get_actor_params() + agent._model.get_critic_params()}
    checks["(a) D's parameters are separate from pi and V"] = not (disc_ids & ppo_ids)
    raw = env.fetch_disc_obs_demo(65536)
    fresh = agent._disc_obs_norm.normalize(raw)
    std = fresh.std(0)
    # constant by construction (raw std ~0, normalizer floor 1e-4): e.g. the 2nd component of the
    # root 6D rotation of s', expressed in its own heading frame. Heavy-tailed joint velocities lose
    # a little std to the +-10 sigma clip: std in [0.9, 1.05] is accepted.
    moving = raw.std(0) > 1e-3
    mean_err = fresh.mean(0).abs().max().item()
    std_min, std_max = std[moving].min().item(), std[moving].max().item()
    checks["(a) normalized mocap: mean ~0, std ~1"] = mean_err < 0.05 and std_min > 0.9 and std_max < 1.05
    print(f"  (a) D separate from pi and V: {not (disc_ids & ppo_ids)}; normalized fresh mocap pairs: "
          f"max |mean| {mean_err:.3f}, std in [{std_min:.3f}, {std_max:.3f}] "
          f"({int((~moving).sum())} constant features)")

    # rollout, as BaseAgent.train_model and _train_iter do
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    agent._init_iter()
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TRAIN)
    with torch.no_grad():
        agent._rollout_train(agent._steps_per_iter)
    buf = agent._exp_buffer
    task_r = buf.get_data_flat("reward").clone()
    disc_obs = buf.get_data_flat("disc_obs").clone()

    # (b) rewards
    agent._build_train_data()
    with torch.no_grad():
        style_r = disc.style_reward(agent._model.eval_disc(agent._disc_obs_norm.normalize(disc_obs)))
    expected = agent._task_reward_weight * task_r + agent._style_reward_weight * style_r
    err_r = (buf.get_data_flat("reward") - expected).abs().max().item()
    checks["(b) reward = w_G r_G + w_S r_S"] = err_r < 1e-5
    print(f"  (b) reward used by GAE vs 0.5 r_G + 0.5 r_S recomputed: max error {err_r:.1e} "
          f"(mean r_G {task_r.mean():.3f}, mean r_S {style_r.mean():.3f}, untrained D)")

    # (c) replay buffer
    n_expected = min(agent._disc_replay_insert, disc_obs.shape[0])
    checks["(c) replay buffer filled"] = len(agent._disc_replay) == n_expected
    print(f"  (c) replay buffer: {len(agent._disc_replay)} pairs (expected {n_expected})")

    # (d) updates
    snap = {n: p.detach().clone() for n, p in agent._model.named_parameters()}
    disc_steps0 = agent._disc_optimizer.get_steps()
    train_info = agent._update_model()
    changed = {n: not torch.equal(p, snap[n]) for n, p in agent._model.named_parameters()}
    pi_v = [changed[n] for n in snap if n.startswith(("_actor", "_action_dist", "_critic")) and "logstd" not in n]
    d = [changed[n] for n in snap if n.startswith("_disc")]
    keys = ["disc_loss", "disc_grad_penalty", "disc_acc_mocap", "disc_acc_agent"]
    finite = all(torch.isfinite(torch.as_tensor(train_info[k])).item() for k in keys)
    steps = agent._disc_optimizer.get_steps() - disc_steps0
    checks["(d) pi, V and D updated, D steps, finite diagnostics"] = any(pi_v) and all(d) and steps == agent._disc_steps and finite
    print(f"  (d) parameters changed: pi and V {sum(pi_v)}/{len(pi_v)}, D {sum(d)}/{len(d)}; "
          f"D optimizer steps {steps} (expected {agent._disc_steps})")
    print("      " + ", ".join(f"{k} {float(train_info[k]):.3f}" for k in keys + ["disc_mocap_mean", "disc_agent_mean"]))

    ok = all(checks.values())
    for name, c in checks.items():
        if not c:
            print(f"  FAIL: {name}")
    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


TESTS = {
    "1": ("discriminator observations of the env", test_disc_obs),
    "2": ("discriminator, eq. 6, 7, 8", test_disc_equations),
    "3": ("AMP agent, one training iteration", test_agent_iteration),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS), help="tests to run (default: all)")
    args = p.parse_args()

    results = {t: TESTS[t][1]() for t in args.tests}
    print("\n=== Summary ===")
    for t, ok in results.items():
        print(f"  test {t:3s} {TESTS[t][0]:45s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
