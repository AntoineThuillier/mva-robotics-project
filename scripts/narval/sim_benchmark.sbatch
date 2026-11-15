#!/bin/bash
# Simulation throughput benchmark on a CPU node of Narval (no GPU needed).
# Submit from the repo root:   sbatch --account=def-XXXX scripts/narval/sim_benchmark.sbatch
#SBATCH --job-name=sim-bench
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

VENV=${VENV:-$HOME/venvs/amp}

module load StdEnv/2023 python/3.11
source "$VENV/bin/activate"

lscpu | grep -E "Model name|^CPU\(s\)"
python scripts/sim_benchmark.py --envs 4096 --threads 1 8 16 32 --sim-hz 120 480
