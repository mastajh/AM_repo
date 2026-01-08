#!/bin/bash
# Visualization script for Car Aerodynamics GNN

set -e

# Default values
MESH=""
RESULTS=""
OUTPUT="outputs/visualizations"
PLOTS="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mesh)
            MESH="$2"
            shift 2
            ;;
        --results)
            RESULTS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --plots)
            PLOTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MESH" ]; then
    echo "Error: --mesh is required"
    exit 1
fi

if [ -z "$RESULTS" ]; then
    echo "Error: --results is required"
    exit 1
fi

echo "========================================="
echo "Car Aerodynamics GNN - Visualization"
echo "========================================="
echo "Mesh: $MESH"
echo "Results: $RESULTS"
echo "Output: $OUTPUT"
echo "Plots: $PLOTS"
echo "========================================="

# Run visualization
python -m main visualize \
    --mesh "$MESH" \
    --results "$RESULTS" \
    --output "$OUTPUT" \
    --plots $PLOTS

echo ""
echo "Visualization complete! Saved to $OUTPUT"
