from vllm import LLM, SamplingParams
import json
from pathlib import Path
from theoremforge.utils import (
    extract_lean_code,
)
from theoremforge.lean_server.server import erase_header, AsyncVerifier
from theoremforge.utils import statement_check
import argparse
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import gc
import asyncio

HEADER = "import Mathlib\n"
formalizer_id = "model/ReForm-32B"
prover_id = "model/Goedel-Prover-V2-32B"

formalizer_prompt = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency.
{problem}
"""

prover_prompt = """Complete the following Lean 4 code:
```lean4
{problem}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""

config = {
    "LocalLeanProject": "/home/yicheng/.lean-cache/mathlib4-v4.19.0",
    "LeanServerWorkers": 8,
    "LeanServerHeader": "import Mathlib\n",
    "LeanServerTimeout": 60,
    "MaxTotalMemory": 0.6,
    "MaxProcessMemory": 0.6,
}
verifier = AsyncVerifier(config)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    print("Loading selected problems...")
    with open(args.input_file, "r") as f:
        problems = [json.loads(line) for line in f]
    print(f"Loaded {len(problems)} problems")

    print("Running formalizer...")
    formalizer = LLM(
        model=formalizer_id, trust_remote_code=True, tensor_parallel_size=2
    )
    formalizer_prompts = [
        formalizer_prompt.format(problem=problem["nl_problem"]) for problem in problems
    ]
    formalizer_sampling_params = SamplingParams(
        temperature=1.0, max_tokens=8192, n=4
    )
    formalizer_results = formalizer.generate(
        formalizer_prompts, formalizer_sampling_params
    )

    formalizer_records = []

    for problem, result in zip(problems, formalizer_results):
        valid_code = None
        for output in result.outputs:
            lean_code = extract_lean_code(output.text)
            if lean_code:
                lean_code = erase_header(lean_code)
            else:
                continue
            verification_result = await verifier.verify(lean_code, True)
            valid = verification_result[0]
            if valid:
                valid_code = lean_code
                break
        if valid_code:
            formalizer_records.append(
                {
                    "id": problem["id"],
                    "nl_problem": problem["nl_problem"],
                    "domain": problem["domain"],
                    "difficulty": problem["difficulty"],
                    "formal_statement": valid_code,
                    "statement_is_valid": True,
                }
            )
        else:
            formalizer_records.append(
                {
                    "id": problem["id"],
                    "nl_problem": problem["nl_problem"],
                    "domain": problem["domain"],
                    "difficulty": problem["difficulty"],
                    "formal_statement": None,
                    "statement_is_valid": False,
                }
            )
    # Log formalizer statistics
    valid_count = sum(record["statement_is_valid"] for record in formalizer_records)
    print(
        f"Formalizer statistics: {valid_count} valid, {len(formalizer_records) - valid_count} invalid"
    )
    destroy_model_parallel()
    del formalizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Formalizer resources released")

    print("Running prover...")
    prover = LLM(model=prover_id, trust_remote_code=True, tensor_parallel_size=2)
    valid_records = [
        record for record in formalizer_records if record["statement_is_valid"]
    ]
    valid_ids = set([record["id"] for record in valid_records])
    prover_prompts = [
        prover_prompt.format(problem=record["formal_statement"])
        for record in valid_records
    ]
    prover_sampling_params = SamplingParams(
        temperature=1.0, max_tokens=8192, n=4
    )
    prover_results = prover.generate(prover_prompts, prover_sampling_params)
    destroy_model_parallel()
    del prover
    gc.collect()
    torch.cuda.empty_cache()
    print("Prover resources released")

    prover_records = {}
    prover_outputs = [[extract_lean_code(output.text) for output in results.outputs] for results in prover_results]
    async def verify_code(statement: str, codes: list[str]):
        codes = [erase_header(code) for code in codes if code is not None]
        codes = [code for code in codes if statement_check(statement, code)]
        verification_result = await verifier.batched_verify(codes, False)
        valid_index = [i for i, result in enumerate(verification_result) if result[0]]
        if valid_index:
            return codes[valid_index[0]]
        return None

    tasks = [verify_code(record["formal_statement"], codes) for record, codes in zip(valid_records, prover_outputs)]
    results = await asyncio.gather(*tasks)
    
    prover_records = {record["id"]: {"formal_proof": result, "proof_is_valid": result is not None} for record, result in zip(valid_records, results)}

    # Log prover statistics
    valid_count = sum(record["proof_is_valid"] for record in prover_records.values())
    print(
        f"Prover statistics: {valid_count} valid, {len(prover_records) - valid_count} invalid"
    )

    final_results = []
    for record in formalizer_records:
        if record["id"] in valid_ids:
            final_results.append(
                {
                    "id": record["id"],
                    "informal_statement": record["nl_problem"],
                    "domain": record["domain"],
                    "difficulty": record["difficulty"],
                    "formal_statement": record["formal_statement"],
                    "formal_proof": prover_records[record["id"]]["formal_proof"],
                    "success": prover_records[record["id"]]["proof_is_valid"],
                }
            )
        else:
            final_results.append(
                {
                    "id": record["id"],
                    "informal_statement": record["nl_problem"],
                    "domain": record["domain"],
                    "difficulty": record["difficulty"],
                    "formal_statement": record["formal_statement"],
                    "formal_proof": None,
                    "success": False,
                }
            )

    with open(results_dir / "expert_results.jsonl", "w") as f:
        for record in final_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    await verifier.close()

if __name__ == "__main__":
    asyncio.run(main())
