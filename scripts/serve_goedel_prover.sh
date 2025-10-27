uv run vllm serve model/Goedel-Prover-V2-32B \
    --dtype auto \
    --port 8001 \
    --tensor-parallel-size 2