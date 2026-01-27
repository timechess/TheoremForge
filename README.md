<div align=center>

# TheoremForge: Scaling up Formal Data Synthesis with Low-Budget Agentic Workflow

[![arXiv](https://img.shields.io/badge/arXiv-2502.14637-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2601.17332)


</div>

TheoremForge is an agentic workflow system for synthesizing formal mathematical data at scale. It combines multiple specialized agents to transform informal mathematical statements into formalized Lean 4 theorems with verified proofs.


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

Edit `config/gemini-3-flash.yaml` to match your setup:

- **ProverAgentConfig**: Points to local vLLM server (see next section)
- **Other agents**: Use [CloseAI](https://referer.shadowai.xyz/r/1038507) or any OpenAI-compatible API

You can modify `base_url` and `api_key` to use alternative providers.

### 6. Model Serving

Download and serve the [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) model and the [ReForm-32B](https://huggingface.co/GuoxinChen/ReForm-32B) model.

#### Download Model

```sh
mkdir -p model
# Download Goedel-Prover-V2-32B and ReForm-32B to model/ directory
# Use huggingface-cli or git lfs
```

#### Serve with vLLM

```sh
bash scripts/vllm_serve_model.sh \
  --model-name model/Goedel-Prover-V2-32B \
  --port 8002 \
  --gpu-ids 0,1

bash scripts/vllm_serve_model.sh \
  --model-name model/ReForm-32B \
  --port 8003 \
  --gpu-ids 2,3
```

Update `config/gemini-3-flash.yaml` with your chosen port and model path.

## Usage

### Workflow Overview

The typical workflow consists of three main steps:

1. **Generate Dataset**: Sample problems from DeepTheorem and DeepMath datasets
2. **Run Workflow**: Process problems through the TheoremForge agentic workflow
3. **Extract Data**: Extract training data for different tasks from workflow results

### Step 1: Generate Dataset

Use `scripts/generate_dataset.py` to sample problems from DeepTheorem and DeepMath datasets by difficulty distribution.

**Example**:
```sh
uv run python scripts/generate_dataset.py \
  --num_samples 1000 \
  --ratio 0.6:0.4 \
  --seed 42 \
  --output data/sampled_problems.jsonl
```

**Parameters**:
- `--num_samples`: Total number of problems to sample (required)
- `--ratio`: Dataset ratio in format `deeptheorem:deepmath`, e.g., `0.6:0.4` (required)
- `--seed`: Random seed (default: 42)
- `--output`: Output file path (optional, prints first sample if not specified)

The script samples problems while preserving the original difficulty distribution from each dataset.

### Step 2: Run Workflow

Use `scripts/run_workflow.py` to process problems through the TheoremForge agentic workflow.

**Example**:
```sh
uv run python scripts/run_workflow.py \
  --config_path config/gemini-3-flash.yaml \
  --max_workers 4 \
  --input_file data/sampled_problems.jsonl \
  --export_file results/workflow_results.jsonl \
  --resume
```

**Parameters**:
- `--config_path`: Path to the configuration file (required)
- `--max_workers`: Maximum number of concurrent workers (required)
- `--input_file`: Input file with problems in JSONL format (required)
  - Each line should be a JSON object with `id` and `nl_problem` fields
- `--export_file`: Output file path for workflow results (required)
- `--resume`: Resume from checkpoint (optional)
  - If specified, resumes from the last successful entry in the export file

**Input Format**:
Each line in the input file should be a JSON object:
```json
{"id": "problem_1", "nl_problem": "Prove that the sum of two even numbers is even."}
```

**Output Format**:
Each line in the output file is a JSON object:
```json
{
  "id": "problem_1",
  "statement_id": "...",
  "formal_statement": "theorem sum_even : ...",
  "informal_statement": "Prove that the sum of two even numbers is even.",
  "formal_proof": "...",
  "success": true
}
```

### Step 3: Extract Data

Use `scripts/extract_data.py` to extract training data for different tasks from workflow results.

**Example**:
```sh
uv run python scripts/extract_data.py \
  --file results/workflow_results.jsonl
```

**Parameters**:
- `--file`: Path to workflow results file (required)

**Output Files**:
The script generates five JSONL files in the `results/` directory:

1. **`statement_formalization_data.jsonl`**: Data for statement formalization task
   - Fields: `informal_statement`, `retrieval_results`, `formal_statement`, `success`

2. **`premise_selection_data.jsonl`**: Data for premise selection task
   - Fields: `informal_statement`/`formal_statement`, `queries`, `results`, `success`

3. **`proof_generation_data.jsonl`**: Data for proof generation task
   - Fields: `formal_statement`, `retrieval_results`, `formal_proof`, `success`

4. **`proof_correction_data.jsonl`**: Data for proof correction task
   - Fields: `error_code`, `error_messages`, `valid_code`, `success`

5. **`proof_sketching_data.jsonl`**: Data for proof sketching task
   - Fields: `formal_statement`, `retrieval_results`, `informal_proof`, `proof_sketch`, `success`

**Prerequisites**:
- The Lean verifier server must be running (see [Step 3: Lean Server](#3-lean-server))
- The database must contain trace information from the workflow run
- Ensure `DATABASE_PATH` in `.env` points to the correct database file

We have open-sourced our extracted data in huggingface: https://huggingface.co/datasets/timechess/theoremforge

## Citation

```bibtex
@misc{tao2026theoremforgescalingformaldata,
      title={TheoremForge: Scaling up Formal Data Synthesis with Low-Budget Agentic Workflow}, 
      author={Yicheng Tao and Hongteng Xu},
      year={2026},
      journal={arXiv preprint arXiv:2601.17332}
}
```