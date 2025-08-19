#!/bin/bash

# Script to train multiple heat tasks with different architectures and random seeds,
# accepting a device ID as input (e.g., ./train_tasks.sh device=1)
# Grid size is 129 for heat-1T* tasks (heat-1T-zsquares, heat-1T-zsquares-t1, heat-1T-zsquares-t1-bmax)
# and 257 for heat-2T* tasks (heat-2T-zsquares, heat-2T-zsquares-t1, heat-2T-zsquares-t1-bmax)

# Configuration
TASKS_FDON=("heat-1T-zsquares-t1" "heat-1T-zsquares-t1-bmax" "heat-2T-zsquares-t1" "heat-2T-zsquares-t1-bmax")
TASKS_FNO=("heat-1T-zsquares" "heat-2T-zsquares")
ARCHS_FDON=("fdon1" "fdon2")
ARCH_FNO="fno"
SEEDS=(0 1 2 3 4)
NUM_TRAIN=600
MODES=12
WIDTH=32
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
    echo "Error: Too many arguments. Expected './train_tasks.sh [device=<number>]'"
    exit 1
else
    echo "No device specified. Using default device: $DEVICE"
fi

# Ensure train.py exists
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in current directory"
    exit 1
fi

# Experiments for fdon1 and fdon2
for TASK in "${TASKS_FDON[@]}"; do
    # Set grid size based on task
    if [[ $TASK == heat-1T* ]]; then
        GRID_SIZE=129
    else
        GRID_SIZE=257
    fi

    for ARCH in "${ARCHS_FDON[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "Starting training for task $TASK with arch $ARCH, seed $SEED, grid-size $GRID_SIZE on device $DEVICE..."

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
                echo "Completed training for task $TASK, arch $ARCH, seed $SEED"
            else
                echo "Error: Training failed for task $TASK, arch $ARCH, seed $SEED"
            fi
            echo "----------------------------------------"
        done
    done
done

# Experiments for fno
for TASK in "${TASKS_FNO[@]}"; do
    # Set grid size based on task
    if [[ $TASK == heat-1T* ]]; then
        GRID_SIZE=129
    else
        GRID_SIZE=257
    fi

    for SEED in "${SEEDS[@]}"; do
        echo "Starting training for task $TASK with arch $ARCH_FNO, seed $SEED, grid-size $GRID_SIZE on device $DEVICE..."

        # Run train.py
        $PYTHON train.py \
            --task "$TASK" \
            --arch "$ARCH_FNO" \
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
            echo "Completed training for task $TASK, arch $ARCH_FNO, seed $SEED"
        else
            echo "Error: Training failed for task $TASK, arch $ARCH_FNO, seed $SEED"
        fi
        echo "----------------------------------------"
    done
done

echo "All training jobs completed."