#!/bin/bash

# Script to train heat-2T-zsquares-t1-bmax task with different architectures,
# snapshot parameters, and random seeds, accepting a device ID as input
# (e.g., ./train_snapshots.sh device=1)

# Configuration
TASK="heat-1T-zsquares-t1-bmax"
ARCHS=("fdon1" "fdon2")
SNAPSHOTS=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
SEEDS=(0 1 2 3 4)
NUM_TRAIN=600
GRID_SIZE=129
MODES=12
WIDTH=32
NUM_BRANCH=4
NUM_TRUNK=2
DATA_ROOT="../dataset/nd_seq"
OUTPUT_ROOT="../result/seq_exps"
PYTHON="python3"  # Adjust if using a specific Python environment
DEFAULT_DEVICE=0  # Default device ID if none provided

# Parse input argument for device
DEVICE=$DEFAULT_DEVICE
if [ $# -eq 1 ]; then
    if [[ $1 =~ ^device=([0-9]+)$ ]]; then
        DEVICE=${BASH_REMATCH[1]}
        echo "Using device: $DEVICE"
    else
        echo "Error: Invalid input format. Expected 'device=<number>', got '$1'"
        exit 1
    fi
elif [ $# -gt 1 ]; then
    echo "Error: Too many arguments. Expected './train_snapshots.sh [device=<number>]'"
    exit 1
else
    echo "No device specified. Using default device: $DEVICE"
fi

# Ensure train.py exists
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in current directory"
    exit 1
fi

# Loop over architectures
for ARCH in "${ARCHS[@]}"; do
    # Loop over snapshots
    for SNAPSHOT in "${SNAPSHOTS[@]}"; do
        # Set dataset path for the current snapshot
        DATASET_PATH="${DATA_ROOT}/${SNAPSHOT}/"
        # Set output directory for the current snapshot
        OUTPUT_DIR="${OUTPUT_ROOT}/${SNAPSHOT}"

        # Check if dataset path exists
        if [ ! -d "$DATASET_PATH" ]; then
            echo "Error: Dataset path $DATASET_PATH does not exist"
            continue
        fi

        # Loop over seeds
        for SEED in "${SEEDS[@]}"; do
            echo "Starting training for task $TASK with arch $ARCH, snapshot $SNAPSHOT, seed $SEED on device $DEVICE..."

            # Run train.py
            $PYTHON train.py \
                --task "$TASK" \
                --arch "$ARCH" \
                --data-root "$DATASET_PATH" \
                --num-train "$NUM_TRAIN" \
                --seed "$SEED" \
                --grid-size "$GRID_SIZE" \
                --modes "$MODES" \
                --width "$WIDTH" \
                --num-branch "$NUM_BRANCH" \
                --num-trunk "$NUM_TRUNK" \
                --output-dir "$OUTPUT_DIR" \
                --device "$DEVICE"

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Completed training for arch $ARCH, snapshot $SNAPSHOT, seed $SEED"
            else
                echo "Error: Training failed for arch $ARCH, snapshot $SNAPSHOT, seed $SEED"
            fi
            echo "----------------------------------------"
        done
    done
done

echo "All training jobs completed."