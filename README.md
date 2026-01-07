# TheoremForge

## Setup

### 1. Clone Repository

```sh
git clone https://github.com/timechess/TheoremForge.git
cd TheoremForge
```

### 2. Python Environment

This project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management.

**Install uv**:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create virtual environment and install dependencies**:

```sh
uv venv
uv sync
```

> **⚠️ vLLM Note**: This project uses vLLM to serve prover models, which may have CUDA compatibility issues. If you encounter problems, run:
> ```sh
> bash scripts/install_vllm.sh
> ```

### 3. Lean Server

The verifier server requires a local Lean 4 installation with Mathlib.

#### Install Lean (Linux)

**Install elan** (Lean version manager):

```sh
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env
```

#### Setup Mathlib4

```sh
git clone https://github.com/leanprover-community/mathlib4.git
cd mathlib4 && git checkout v4.21.0

# Build the project (this may take a while)
lake exe cache get
lake build
```

You should see `Build completed successfully` when finished.

#### Start the Verifier Server

After completing [Configuration](#5-configuration), start the server:

```sh
# The server is on port 8000
uv run run_lean_server
```

### 4. Search Server

This project uses [LeanExplore](https://www.leanexplore.com/) as the search service. Run the following command to download cache files.

```sh
uv run leanexplore data fetch
```

### 5. Configuration

#### Environment Variables

Create a `.env` file in the project root:

```sh
CLOSEAI_API_KEY=YOUR_CLOSEAI_API_KEY
DATABASE_PATH=./theoremforge.db
```

#### Configuration File

Edit `config.yaml` to match your setup:

- **ProverAgentConfig**: Points to local vLLM server (see next section)
- **Other agents**: Use [CloseAI](https://referer.shadowai.xyz/r/1038507) or any OpenAI-compatible API

You can modify `base_url` and `api_key` to use alternative providers.

### 6. Model Serving

Download and serve the [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) model and the [ReForm-8B](https://huggingface.co/GuoxinChen/ReForm-8B) model.

#### Download Model

```sh
mkdir -p model
# Download Goedel-Prover-V2-32B and ReForm-8B to model/ directory
# Use huggingface-cli or git lfs
```

#### Serve with vLLM

```sh
bash scripts/vllm_serve_model.sh \
  --model-name model/Goedel-Prover-V2-32B \
  --port 8002 \
  --gpu-ids 0,1

bash scripts/vllm_serve_model.sh \
  --model-name model/ReForm-8B \
  --port 8003 \
  --gpu-ids 2,3
```

Update `config/gemini-3-flash.yaml` with your chosen port and model path.

