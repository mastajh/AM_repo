#!/bin/bash
# Training script for Car Aerodynamics GNN

set -e

# Default values
CONFIG="configs/ahmed_body.yaml"
NUM_WORKERS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --cpu)
            USE_CPU="--cpu"
            shift
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Car Aerodynamics GNN - Training"
echo "========================================="
echo "Config: $CONFIG"
echo "Workers: $NUM_WORKERS"

# Build command
CMD="python -m main train --config $CONFIG --num_workers $NUM_WORKERS"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
    echo "Resume: $RESUME"
fi

if [ -n "$USE_CPU" ]; then
    CMD="$CMD --cpu"
    echo "Device: CPU"
else
    echo "Device: GPU"
fi

echo "========================================="

# Run training
$CMD
