"""TD(lambda) returns, from which the GAE advantages follow (replaces MimicKit's learning/rl_util.py).

    R_t = r_t + gamma * ((1 - lambda_t) V(s_{t+1}) + lambda_t R_{t+1}),   lambda_t = lambda (0 if the
          episode ended at step t)
    advantage A_t = R_t - V(s_t)       (= GAE(lambda), Schulman et al. 2015)

At the last step of the buffer, R_T = r_T + gamma V(s_{T+1}) (bootstrap). When an episode ends at
step t, V(s_{t+1}) is the value of the end: 0-reward fall (FAIL) -> r_fail / (1 - gamma), reached
goal (SUCC) -> r_succ / (1 - gamma); time limit (TIME) keeps V(s_{t+1}), the episode was only cut.
"""

import torch

# environment's done flags (MimicKit's envs/base_env.py DoneFlags)
DONE_NULL, DONE_FAIL, DONE_SUCC, DONE_TIME = 0, 1, 2, 3


def end_values(next_vals, done, discount, reward_fail, reward_succ):
    """V(s_{t+1}) with the value of the episode end where it ended in a fall or a success."""
    next_vals = next_vals.clone()
    next_vals[done == DONE_SUCC] = reward_succ / (1.0 - discount)
    next_vals[done == DONE_FAIL] = reward_fail / (1.0 - discount)
    return next_vals


def td_lambda_returns(r, next_vals, done, discount, td_lambda):
    """r, next_vals, done: (T, N). Returns R (T, N)."""
    ret = torch.zeros_like(r)
    ended = (done != DONE_NULL).float()
    ret[-1] = r[-1] + discount * next_vals[-1]
    for t in reversed(range(r.shape[0] - 1)):
        lam = td_lambda * (1.0 - ended[t])
        ret[t] = r[t] + discount * ((1.0 - lam) * next_vals[t] + lam * ret[t + 1])
    return ret
