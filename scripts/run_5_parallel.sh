#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Ctrl+C pressed, killing all running trials..."; pkill -P $$; exit 1' SIGINT

MAX_JOBS=5

EVAL_DIR="/mnt/flex-s-pypzpbfqm6/fitam/results/evaluation_requests/all_test_counties"
SAVE_ROOT="/mnt/flex-s-pypzpbfqm6/fitam/results/diffusion_results"

SWATH_LIB="/mnt/flex-s-pypzpbfqm6/fitam/results/swaths/simulated_radial_configs/diffusion_radial_map_config.pkl"
RADIAL_CFG="/mnt/flex-s-pypzpbfqm6/fitam/results/configs/simulated_radial_configs/diffusion_radial_map_config.json"
MODEL="/mnt/flex-s-pypzpbfqm6/fitam/results/models/model-50-128-128-lcm-palette.pt"
TRAIN_CFG="/mnt/flex-s-pypzpbfqm6/fitam/results/configs/train_config_dino_classification.json"
DATASET_CFG="/mnt/flex-s-pypzpbfqm6/fitam/results/configs/classification_dataset_config.json"
LOGGING_CFG="/mnt/flex-s-pypzpbfqm6/fitam/results/configs/logging_config.json"
EVAL_CFG="/mnt/flex-s-pypzpbfqm6/fitam/results/configs/evaluation_config.json"

run_trial () {
    local eval_json="$1"
    echo "Starting: $eval_json"

    python src/fitam/evaluation/planner_trial_opengl.py \
        --eval_request_path "$eval_json" \
        --save_root "$SAVE_ROOT" \
        --job_numbers 0 \
        --swath_library_path "$SWATH_LIB" \
        --radial_costmap_config_path "$RADIAL_CFG" \
        --model_path "$MODEL" \
        --training_config_path "$TRAIN_CFG" \
        --dataset_config_path "$DATASET_CFG" \
        --logging_config_path "$LOGGING_CFG" \
        --eval_config_path "$EVAL_CFG" \
        --dump_all_outputs
}

export -f run_trial

job_count=0

for eval_json in "$EVAL_DIR"/*.json; do
    map_name=$(basename "$eval_json" .json)
    out_dir="$SAVE_ROOT/$map_name/0000000"
    echo "OUT DIR: $out_dir"
    if [[ -d "$out_dir" ]]; then
        echo "Skipping (already exists): $map_name/0000000"
        continue
    fi

    
    python src/fitam/evaluation/planner_trial_opengl.py \
        --eval_request_path "$eval_json" \
        --save_root "$SAVE_ROOT/$map_name" \
        --job_numbers 0 \
        --swath_library_path "$SWATH_LIB" \
        --radial_costmap_config_path "$RADIAL_CFG" \
        --model_path "$MODEL" \
        --training_config_path "$TRAIN_CFG" \
        --dataset_config_path "$DATASET_CFG" \
        --logging_config_path "$LOGGING_CFG" \
        --eval_config_path "$EVAL_CFG" \
        --dump_all_outputs &
	#run_trial "$eval_json" &

    ((job_count+=1))
    echo "JOB_COUNT: $job_count"

    if [[ "$job_count" -ge "$MAX_JOBS" ]]; then
        wait -n
        ((job_count-=1))
    fi
done

wait
echo "All trials complete ✅"

