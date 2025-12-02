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

The search server provides semantic retrieval of relevant Mathlib theorems using Qdrant vector database.

#### Prerequisites

- Docker and Docker Compose
- Complete the [Configuration](#5-configuration) section first

#### Setup Vector Database

**Start Docker containers**:

```sh
docker compose up -d
```

**Upload theorem embeddings** (using `all-mpnet-base-v2`):

```sh
# Multi-GPU (recommended for faster processing)
uv run python scripts/upload_const.py \
  --dataset_name FrenzyMath/mathlib_informal_v4.19.0 \
  --gpu_ids 0,1,2,3

# Single GPU
uv run python scripts/upload_const.py \
  --dataset_name FrenzyMath/mathlib_informal_v4.19.0 \
  --no_multi_gpu
```

This process embeds ~100K informal theorem descriptions from the [Mathlib dataset](https://huggingface.co/datasets/FrenzyMath/mathlib_informal_v4.19.0).

#### Start the Search Server

```sh
uv run python scripts/run_search_server.py --port 8001
```

#### Test the Search Server

```sh
curl --request POST \
  --url http://localhost:8001/search \
  --header 'Content-Type: application/json' \
  --data '{
    "queries": ["Prime number"],
    "topk": 5
  }' | jq
```

### 5. Configuration

#### Environment Variables

Create a `.env` file in the project root:

```sh
CLOSEAI_API_KEY=YOUR_CLOSEAI_API_KEY
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
MONGO_PASSWORD=YOUR_MONGODB_PASSWORD
DATABASE_URL=mongodb://admin:${MONGO_PASSWORD}@localhost:27018/theoremforge?authSource=admin
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

Update `config.yaml` with your chosen port and model path.

---

## 比赛结果复现

在完成上述配置后，使用 `main.py` 脚本产出比赛解答。赛题文件位于 `competition_problem` 目录下，目前脚本仅支持单文件输入输出，例如：

```sh
uv run python main.py --dataset_path competition_problem/lean_1106.jsonl --output_file 1106.jsonl
```

每日的初赛解答文件位于 `competition_result` 目录下。

**Warning**：由于模型输出的随机性，部分题目经过了多次运行才成功解答，不保证一次性能成功得到与给出的解答相同的结果。`config.yaml` 中为大部分题目运行时使用的模型，但不排除部分题目更换了其中的部分模型的情况。