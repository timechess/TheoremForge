# TheoremForge

## Setup

First clone this repository:

```sh
git clone https://github.com/timechess/TheoremForge.git
cd TheoremForge
```

### Python Environment

This project uses `uv` to manage python dependencies. Install uv via:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now create virtual environment and install dependencies using:

```sh
uv venv
uv sync
```

**vLLM Note**: This project uses vLLM to serve prover models, which may have dependency issues due to different cuda versions. You can run the command below to override the installation above:

```sh
bash scripts/install_vllm.sh
```

### Lean Server

This project depends on a local Lean project to setup a verifier server. You should first install Lean on your device. Here is the instruction for Linux:

First install `elan`:

```sh
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Make sure to add the environment variables:

```sh
source $HOME/.elan/env
```

Setup `mathlib4` as our local project (the environment for the verifier server):

```sh
git clone https://github.com/leanprover-community/mathlib4.git

# Checkout your specified revision. Here we use v4.21.0
cd mathlib4 && git checkout v4.21.0

# Build the project
lake exe cache get
lake build
```

If everything goes well, you should see `Build completed successfully` at the end of the output. Before starting the verifier server, follow [Configuration and Environment Variables](#configuration-and-environment-variables). Then, run the following command to start the verifier server:

```sh
uv run run_lean_server
```

### Search Server

This project also needs a semantic search engine for Mathlib theorems. We use the open-sourced [dataset](https://huggingface.co/datasets/FrenzyMath/mathlib_informal_v4.19.0) which contains informal descriptions of the theorems. We use [Qdrant](https://qdrant.tech/) as the vector database, setup by docker. You should first configure the environment variables in `.env` before running `docker compose up -d` to start the service. For the instructions to setup environment variables, see [Configuration and Environment Variables](#configuration-and-environment-variables).

After start the docker containers, run the following command to encode and upload the embeddings of the informal descriptions to the vector database. We use `all-mpnet-base-v2` as the embedding model for efficiency.

```sh
# Use multi-gpus (4 as an example)
uv run python scripts/upload_const.py --dataset_name FrenzyMath/mathlib_informal_v4.19.0 --gpu_ids 0,1,2,3

# Not use multi-gpus
uv run python scripts/upload_const.py --dataset_name FrenzyMath/mathlib_informal_v4.19.0 --no_multi_gpu
```

This process may take some time depending on your hardware. Then start the server by running:

```sh
uv run python scripts/run_search_server.py --port 8002
```

Run the following command to test:

```sh
curl --request POST \
  --url http://localhost:8002/search \
  --header 'Content-Type: application/json' \
  --data '{
    "queries": ["Prime number"],
    "topk": 5
}' | jq
```

### Configuration and Environment Variables

Your `.env` file should contain:

```sh
CLOSEAI_API_KEY=YOUR_CLOSEAI_API_KEY
MONGO_PASSWORD=YOUR_MONGODB_PASSWORD
DATABASE_URL=mongodb://admin:${MONGO_PASSWORD}@localhost:27018/theoremforge?authSource=admin
```

After setup the variables, run `docker compose up -d` to start the docker containers.

The configuration file is `config.yaml`. Note that `ProverAgentConfig` depends on local vLLM server, the others depend on [CloseAI](https://referer.shadowai.xyz/r/1038507) (This is my invitation link). You can also use other model API providers by changing the `base_url`, but you don't need to modify the name of the environment variable `CLOSEAI_API_KEY`.

> TODO: Support using different model providers.

Create a directory named `model` and download [Goedel-Prover-V2](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) to this directory. Run the following command to serve the prover model.

```sh
bash scripts/serve_goedel_prover.sh model/Goedel-Prover-V2-32B ${PROVER_PORT} ${TENSOR_PARALLEL_SIZE}
# Example: bash scripts/serve_goedel_prover.sh model/Goedel-Prover-V2-32B 8001 4
```

Modify the `config.yaml` according to your port and model.

After setting up the 3 servers (verifier server, search server and prover model), you can start to run the agent workflow.

## Example Usage

Run the workflow on `minif2f_test` using this script:

```sh
uv run python main.py
```