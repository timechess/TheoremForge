#!/bin/bash

# Script to serve a model using vLLM with named arguments
# Usage: ./vllm_serve_model.sh --model-name <path> --port <port> --gpu-ids <gpu_ids>
# Example: ./vllm_serve_model.sh --model-name "meta-llama/Llama-2-7b" --port 8000 --gpu-ids "0,1,2,3"

set -e

# Default values
MODEL_NAME=""
PORT=""
GPU_IDS=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-name <path> --port <port> --gpu-ids <gpu_ids>"
            echo ""
            echo "Arguments:"
            echo "  --model-name    Path to the model"
            echo "  --port          Port to serve the model"
            echo "  --gpu-ids       Comma-separated GPU IDs (e.g., '0,1,2,3')"
            echo ""
            echo "Example:"
            echo "  $0 --model-name 'meta-llama/Llama-2-7b' --port 8000 --gpu-ids '0,1,2,3'"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model-name is required"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: --port is required"
    exit 1
fi

if [ -z "$GPU_IDS" ]; then
    echo "Error: --gpu-ids is required"
    exit 1
fi

# Calculate tensor parallel size from GPU IDs
# Count the number of comma-separated GPU IDs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "Starting vLLM server with the following configuration:"
echo "  Model: $MODEL_NAME"
echo "  Port: $PORT"
echo "  GPU IDs: $GPU_IDS"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Run vLLM
uv run vllm serve "$MODEL_NAME" \
    --dtype auto \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"