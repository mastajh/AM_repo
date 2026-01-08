#!/bin/bash
# Inference script for Car Aerodynamics GNN

set -e

# Default values
CONFIG="configs/ahmed_body.yaml"
CHECKPOINT="checkpoints/best_model.pt"
INPUT=""
OUTPUT="outputs/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --cpu)
            USE_CPU="--cpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT" ]; then
    echo "Error: --input is required"
    exit 1
fi

echo "========================================="
echo "Car Aerodynamics GNN - Inference"
echo "========================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Build command
CMD="python -m main inference --config $CONFIG --checkpoint $CHECKPOINT --input $INPUT --output $OUTPUT"

if [ -n "$USE_CPU" ]; then
    CMD="$CMD --cpu"
    echo "Device: CPU"
else
    echo "Device: GPU"
fi

echo "========================================="

# Run inference
$CMD

echo ""
echo "Inference complete! Results saved to $OUTPUT"
