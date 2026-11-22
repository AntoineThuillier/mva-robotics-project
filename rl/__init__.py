"""Our own reinforcement learning code: networks, PPO, GAE, training loop (and AMP on top).

Replaces MimicKit's `mimickit/learning/`; from MimicKit we only keep the "world" (Newton
engine, environments, MotionLib, skeleton). First goal: reproduce MimicKit's PPO exactly
(same settings, same details), checked brick by brick by validation/rl_check.py.
"""
