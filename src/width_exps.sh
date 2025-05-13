#!/bin/bash

# Script to train heat-2T-zsquares-t1-bmax task with different architectures,
# network widths, and random seeds, accepting a device ID as input
# (e.g., ./train_widths.sh device=1)

# Configuration
TASK="heat-2T-zsquares-t1-bmax"
ARCHS=("fdon1" "fdon2")
WIDTHS=(8 16 32)
SEEDS=(0 1 2 3 4)
NUM_TRAIN=600
GRID_SIZE=257
MODES=12
NUM_BRANCH=4
NUM_TRUNK=2
RATIO=1
OUTPUT_DIR="../result/exps"
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
    echo "Error: Too many arguments. Expected './train_widths.sh [device=<number>]'"
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
    # Loop over widths
    for WIDTH in "${WIDTHS[@]}"; do
        # Loop over seeds
        for SEED in "${SEEDS[@]}"; do
            echo "Starting training for task $TASK with arch $ARCH, width $WIDTH, seed $SEED on device $DEVICE..."

            # Run train.py
            $PYTHON train.py \
                --task "$TASK" \
                --arch "$ARCH" \
                --ratio "$RATIO" \
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
                echo "Completed training for arch $ARCH, width $WIDTH, seed $SEED"
            else
                echo "Error: Training failed for arch $ARCH, width $WIDTH, seed $SEED"
            fi
            echo "----------------------------------------"
        done
    done
done

echo "All training jobs completed."