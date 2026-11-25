# Script to lauchn the runs and evaluations

set -e
cd "$(dirname "$0")"
mkdir -p outputs/rl

# RUN
# run <name> <rl/train.py arguments>: training, then evaluation and videos
run() {
    name=$1; shift
    echo "$(date +%H:%M) start $name"
    python -u rl/train.py "$@" --out_dir "outputs/rl/$name" > "outputs/rl/$name.out" 2>&1
    echo "$(date +%H:%M) end   $name"
    evaluate "$name"
}

# EVALUATE
# evaluate <name>: eval_policy and video_policy on outputs/rl/<name> (failures reported, not fatal)
evaluate() {
    name=$1
    dir=outputs/rl/$name
    echo "$(date +%H:%M) eval  $name"
    python -u validation/eval_policy.py --run_dir "$dir" > "$dir/eval_policy.txt" 2>&1 \
        || echo "      eval_policy: some tests FAIL or error, see $dir/eval_policy.txt"
    { python -u validation/video_policy.py --run_dir "$dir" \
        && python -u validation/video_policy.py --run_dir "$dir" --fixed_goal; } > "$dir/videos.out" 2>&1 \
        || echo "      video_policy: error, see $dir/videos.out"
}

# AMP
amp() {
    task=$1; series=$2
    name=amp_${task}_${series}
    [ "$series" = 1 ] && name=amp_${task}
    run "$name" --agent_config "configs/run$series/amp_agent.yaml" --env_config "configs/run$series/${task}_env.yaml"
}

if [ "$1" = eval ]; then
    shift
    for name in "$@"; do evaluate "$name"; done
    exit 0
fi

# ======== queue ========
# the two fixes of the Run neck (series 5 and 6)
amp locomotion 6

# to complete series 2 and 4 (amp_walk_2 and amp_run_4 were stopped: their folders are overwritten)
amp walk 2
amp locomotion 2
amp run 4
amp walk 4
amp locomotion 4

echo "$(date +%H:%M) all runs done"
