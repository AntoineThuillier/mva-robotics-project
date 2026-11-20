"""Train or test a policy on our environments with MimicKit's run.py (phase 3 and later).

    python amp/run.py --arg_file configs/ppo_heading_humanoid_args.txt
    python amp/run.py --arg_file configs/ppo_heading_humanoid_args.txt --max_samples 2000000 --out_dir outputs/phase3/smoke
    python amp/run.py --arg_file configs/ppo_heading_humanoid_args.txt --mode test --num_envs 32 \
        --model_file outputs/phase3/ppo_heading/model.pt --visualize true

MimicKit's env_builder only knows MimicKit's environments: this script adds ours
("task_heading", amp/heading_env.py), then calls MimicKit's main. Everything else (PPO,
logs, saved models, test mode) is MimicKit's.

Paths: MimicKit reads its files relative to its own root, where this script moves. Paths of
the arguments below are resolved against the project root when the file exists there (and
always for --out_dir), otherwise left relative to MimicKit's root. Arguments given on the
command line override those of --arg_file.

Run from the project root, with the project venv (.venv).
"""

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MIMICKIT = PROJECT / "reference" / "MimicKit"
PATH_ARGS = ("env_config", "engine_config", "agent_config", "model_file", "out_dir")


def read_arg_file(path):
    """Tokens of a MimicKit argument file (comments start with #)."""
    tokens = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens += line.split()
    return tokens


def resolve_paths(tokens):
    """Makes the path arguments absolute when they are relative to the project root."""
    out, key = [], None
    for tok in tokens:
        if tok.startswith("--"):
            key = tok[2:]
        elif key in PATH_ARGS and not Path(tok).is_absolute():
            p = PROJECT / tok
            if key == "out_dir" or p.exists():
                tok = str(p)
        out.append(tok)
    return out


def build_argv(argv):
    """Command line first (MimicKit keeps the first value of a key), then the --arg_file tokens."""
    cli = list(argv[1:])
    arg_file = None
    if "--arg_file" in cli:
        i = cli.index("--arg_file")
        arg_file = cli[i + 1]
        del cli[i:i + 2]
    tokens = resolve_paths(cli)
    if arg_file is not None:
        p = Path(arg_file) if Path(arg_file).is_absolute() else PROJECT / arg_file
        tokens += resolve_paths(read_arg_file(p))
    return [argv[0]] + tokens


def setup():
    os.chdir(MIMICKIT)
    for p in (str(PROJECT), str(MIMICKIT / "mimickit")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("WARP_CACHE_PATH", str(PROJECT / "reference" / "warpcache"))

    import envs.env_builder as env_builder
    from amp.heading_env import build_heading_env

    mimickit_build_env = env_builder.build_env

    def build_env(env_file, engine_file, num_envs, device, visualize, record_video=False):
        if env_builder.load_config(env_file)["env_name"] == "task_heading":
            return build_heading_env(num_envs, device, env_file=env_file, engine_file=engine_file,
                                     visualize=visualize, record_video=record_video)
        return mimickit_build_env(env_file, engine_file, num_envs, device, visualize, record_video)

    env_builder.build_env = build_env


if __name__ == "__main__":
    argv = build_argv(sys.argv)
    setup()
    # MimicKit's run.py: reference/MimicKit/mimickit is first in sys.path, so "run" is MimicKit's
    # file, not this one (which runs as __main__)
    from run import main
    main(argv)
