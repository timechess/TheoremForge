#!/bin/bash

# $1: The path to the model
# $2: The port to serve the model
# $3: The number of tensor parallel size

uv run vllm serve $1 \
    --dtype auto \
    --port $2 \
    --tensor-parallel-size $3