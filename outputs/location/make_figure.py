"""Comparison figure of the 4 Target Location runs: outputs/location/location_comparison.png.
Reads outputs/rl/<run>/log.txt and outputs/rl/<run>/eval.txt (validation/eval_location.py output).
    .venv/bin/python outputs/location/make_figure.py"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"amp_location_locomotion": "AMP Locomotion", "amp_location_walk": "AMP Walk",
        "amp_location_run": "AMP Run", "ppo_location": "PPO only (no data)"}
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
for r, lab in RUNS.items():
    L = [l.split() for l in open(ROOT / f"outputs/rl/{r}/log.txt") if l.strip()]
    ix = {k: i for i, k in enumerate(L[0])}
    rows = [list(map(float, x)) for x in L[1:]]
    a.plot([x[ix["Samples"]] / 1e6 for x in rows], [x[ix["Test_Return"]] / 600 for x in rows], label=lab)
    line = re.search(r"0-1 m:.*", open(ROOT / f"outputs/rl/{r}/eval.txt").read()).group(0)
    vals = [float(v) for v in re.findall(r"m: (-?[\d.]+)", line)]
    b.plot([i + 0.5 for i in range(len(vals))], vals, "o-", label=lab)
a.set_xlabel("samples (M)"); a.set_ylabel("test return / 600 steps")
a.set_title("Learning curves (task reward r_G per step, test episodes)")
a.grid(alpha=0.3); a.legend(fontsize=8)
b.set_xlabel("remaining distance to the target (m)"); b.set_ylabel("median COM speed towards target (m/s)")
b.set_title("Speed vs distance (fixed targets at 2-10 m)")
b.axhline(1, color="gray", ls="--", lw=0.8); b.grid(alpha=0.3); b.legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "location_comparison.png", dpi=120)
