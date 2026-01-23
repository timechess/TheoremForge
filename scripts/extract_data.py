import asyncio
import json
import argparse
import os
from theoremforge.db import SQLiteClient
from theoremforge.utils import extract_lean_code, statement_check
from theoremforge.retriever import Retriever
from theoremforge.lean_server.client import RemoteVerifier
import re
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def extract_search_queries(text: str):
    pattern = r"<search>(.*?)</search>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches][:5]


def extract_definitions(text: str):
    pattern = r"<definition>(.*?)</definition>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_theorems(text: str):
    pattern = r"<theorem>(.*?)</theorem>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_informal_proof(text: str) -> str | None:
    pattern = r"<informal_proof>(.*?)</informal_proof>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


async def extract_statement_formalization(results, db: SQLiteClient):
    print("Start to extract Statement Formalization data")
    data = []
    for result in tqdm(results):
        if not result["formal_statement"]:
            continue
        traces = await db.find_traces(
            {
                "state_id": result["statement_id"],
                "agent_name": "definition_retrieval_agent",
            }
        )
        traces.sort(key=lambda trace: trace["created_at"], reverse=True)
        definitions = extract_definitions(traces[0]["response"][0])
        data.append(
            {
                "informal_statement": result["informal_statement"],
                "retrieval_results": definitions,
                "formal_statement": result["formal_statement"],
                "success": result["success"],
            }
        )
    return data


async def extract_premise_selection_data(
    results, db: SQLiteClient, retriever: Retriever
):
    print("Start to extract Premise Selection data")
    data = []
    for result in tqdm(results):
        if result["formal_statement"]:
            definition_retrieval_traces = await db.find_traces(
                {
                    "state_id": result["statement_id"],
                    "agent_name": "definition_retrieval_agent",
                }
            )
            definition_retrieval_traces.sort(
                key=lambda trace: trace["created_at"], reverse=True
            )
            queries = extract_search_queries(
                definition_retrieval_traces[1]["response"][0]
            )
            query_results = await retriever.search_async(queries, 5)
            definitions = [
                result.primary_declaration.lean_name for result in query_results
            ]
            definition_query_map = {}
            for i, query in enumerate(queries):
                for j in range(5):
                    definition_query_map[definitions[5 * i + j]] = (
                        definition_query_map.get(definitions[5 * i + j], []) + [query]
                    )
            valid_definitions = list(
                set(
                    [
                        definition
                        for definition in definitions
                        if definition in result["formal_statement"]
                    ]
                )
            )
            valid_queries = list(
                set(
                    sum(
                        [
                            definition_query_map[definition]
                            for definition in valid_definitions
                        ],
                        [],
                    )
                )
            )
            if valid_queries:
                data.append(
                    {
                        "informal_statement": result["informal_statement"],
                        "formal_statement": "",
                        "queries": valid_queries,
                        "results": valid_definitions,
                        "success": result["success"],
                    }
                )
        theorem_retrieval_traces = await db.find_traces(
            {
                "state_id": result["statement_id"],
                "agent_name": "theorem_retrieval_agent",
            }
        )
        if theorem_retrieval_traces:
            theorem_retrieval_traces.sort(
                key=lambda trace: trace["created_at"], reverse=True
            )
            queries = extract_search_queries(theorem_retrieval_traces[1]["response"][0])
            query_results = await retriever.search_async(queries, 5)
            theorems = [
                result.primary_declaration.lean_name for result in query_results
            ]
            theorem_query_map = {}
            for i, query in enumerate(queries):
                for j in range(5):
                    theorem_query_map[theorems[5 * i + j]] = theorem_query_map.get(
                        theorems[5 * i + j], []
                    ) + [query]
            valid_theorems = []
            if result["formal_proof"]:
                valid_theorems.extend(
                    [
                        theorem
                        for theorem in theorems
                        if theorem in result["formal_proof"]
                    ]
                )
            state = await db.get_state(result["statement_id"])
            if state["subgoals"]:
                for subgoal in state["subgoals"]:
                    substate = await db.get_state(subgoal)
                    if substate["formal_proof"]:
                        valid_theorems.extend(
                            [
                                theorem
                                for theorem in theorems
                                if theorem in substate["formal_proof"]
                            ]
                        )
            valid_theorems = list(set(valid_theorems))
            valid_queries = list(
                set(
                    sum(
                        [theorem_query_map[theorem] for theorem in valid_theorems],
                        [],
                    )
                )
            )
            if valid_queries:
                data.append(
                    {
                        "informal_statement": "",
                        "formal_statement": result["formal_statement"],
                        "queries": valid_queries,
                        "results": valid_theorems,
                        "success": result["success"],
                    }
                )
    return data


async def extract_proof_generation_data(results, db: SQLiteClient):
    print("Start to extract Proof Generation data")
    data = []
    for result in tqdm(results):
        retrieval_traces = await db.find_traces(
            {
                "state_id": result["statement_id"],
                "agent_name": "theorem_retrieval_agent",
            }
        )
        if not retrieval_traces:
            theorems = []
        else:
            retrieval_traces.sort(key=lambda trace: trace["created_at"], reverse=True)
            theorems = extract_theorems(retrieval_traces[0]["response"][0])
        if result["success"]:
            data.append(
                {
                    "formal_statement": result["formal_statement"],
                    "retrieval_results": theorems,
                    "formal_proof": result["formal_proof"],
                    "success": result["success"],
                }
            )
        state = await db.get_state(result["statement_id"])
        if state["subgoals"]:
            for subgoal in state["subgoals"]:
                substate = await db.get_state(subgoal)
                if substate["success"] and statement_check(substate["formal_statement"], substate["formal_proof"]):
                    data.append(
                        {
                            "formal_statement": substate["formal_statement"],
                            "retrieval_results": theorems,
                            "formal_proof": substate["formal_proof"],
                            "success": result["success"],
                        }
                    )
    return data


async def extract_proof_correction_data(results, db: SQLiteClient):
    print("Start to extract Proof Correction data")
    verifier = RemoteVerifier("http://localhost:8000")
    data = []
    proof_correction_error_pattern = (
        r"Compilation Error Message:\s*(.*?)\s*Instructions:"
    )
    proof_correction_code_pattern = r"Lean Code:\s*(.*?)\s*Compilation Error Message:"
    shallow_refinement_error_pattern = r"Compilation Error:\s*(.*?)\s*Useful Theorems:"
    shallow_refinement_code_pattern = r"Failed Proof:\s*(.*?)\s*Compilation Error:"
    for result in tqdm(results):
        if result["success"]:
            sketch_trace = await db.find_traces(
                {"state_id": result["statement_id"], "agent_name": "proof_sketch_agent"}
            )
            proof_correction_traces = await db.find_traces(
                {
                    "state_id": result["statement_id"],
                    "agent_name": "proof_correction_agent",
                }
            )
            if not sketch_trace and proof_correction_traces:
                for proof_correction_trace in proof_correction_traces:
                    error_messages = (
                        re.search(
                            proof_correction_error_pattern,
                            json.loads(proof_correction_trace["prompt"])[0]["content"],
                            re.DOTALL,
                        )
                        .group(1)
                        .strip("\n")
                    )
                    error_code = extract_lean_code(
                        re.search(
                            proof_correction_code_pattern,
                            json.loads(proof_correction_trace["prompt"])[0]["content"],
                            re.DOTALL,
                        ).group(1)
                    )
                    valid_code = extract_lean_code(proof_correction_trace["response"][0])
                    if not statement_check(result["formal_statement"], valid_code):
                        continue
                    valid, _, _ = await verifier.verify(valid_code, False)
                    if valid:
                        data.append(
                            {
                                "error_code": error_code,
                                "error_messages": error_messages,
                                "valid_code": valid_code,
                                "success": result["success"],
                            }
                        )

        state = await db.get_state(result["statement_id"])
        if state["subgoals"]:
            for subgoal in state["subgoals"]:
                substate = await db.get_state(subgoal)
                if substate["success"] and statement_check(substate["formal_statement"], substate["formal_proof"]):
                    shallow_traces = await db.find_traces(
                        {"state_id": subgoal, "agent_name": "shallow_solve_agent"}
                    )
                    proof_correction_traces = await db.find_traces(
                        {"state_id": subgoal, "agent_name": "proof_correction_agent"}
                    )
                    if not shallow_traces and proof_correction_traces:
                        for proof_correction_trace in proof_correction_traces:
                            error_messages = (
                                re.search(
                                    proof_correction_error_pattern,
                                    json.loads(proof_correction_trace["prompt"])[0][
                                        "content"
                                    ],
                                    re.DOTALL,
                                )
                                .group(1)
                                .strip("\n")
                            )
                            error_code = extract_lean_code(
                                re.search(
                                    proof_correction_code_pattern,
                                    json.loads(proof_correction_trace["prompt"])[0][
                                        "content"
                                    ],
                                    re.DOTALL,
                                ).group(1)
                            )
                            valid_code = extract_lean_code(
                                proof_correction_trace["response"][0]
                            )
                            if not statement_check(substate["formal_statement"], valid_code):
                                continue
                            valid, _, _ = await verifier.verify(valid_code, False)
                            if valid:
                                data.append(
                                    {
                                        "error_code": error_code,
                                        "error_messages": error_messages,
                                        "valid_code": valid_code,
                                        "success": result["success"],
                                    }
                                )

                    if len(shallow_traces) > 1:
                        shallow_traces.sort(
                            key=lambda trace: trace["created_at"], reverse=True
                        )
                        error_messages = re.search(
                            shallow_refinement_error_pattern,
                            json.loads(shallow_traces[0]["prompt"])[0]["content"],
                            re.DOTALL,
                        ).group(1)
                        error_code = extract_lean_code(
                            re.search(
                                shallow_refinement_code_pattern,
                                json.loads(shallow_traces[0]["prompt"])[0]["content"],
                                re.DOTALL,
                            ).group(1)
                        )
                        valid_code = extract_lean_code(shallow_traces[0]["response"][0])
                        if not statement_check(substate["formal_statement"], valid_code):
                            continue
                        valid, _, _ = await verifier.verify(valid_code, False)
                        if valid:
                            data.append(
                                {
                                    "error_code": error_code,
                                    "error_messages": error_messages,
                                    "valid_code": valid_code,
                                    "success": result["success"],
                                }
                            )
                        continue
    return data


async def extract_proof_sketching_data(results, db: SQLiteClient):
    print("Start to extract Proof Sketching data")
    data = []
    for result in tqdm(results):
        subgoal_extraction_trace = await db.find_traces(
            {
                "state_id": result["statement_id"],
                "agent_name": "subgoal_extraction_agent",
            }
        )
        if subgoal_extraction_trace:
            sketch = extract_lean_code(
                json.loads(subgoal_extraction_trace[0]["prompt"])[0]["content"]
            )
            informal_proof_trace = await db.find_traces(
                {
                    "state_id": result["statement_id"],
                    "agent_name": "informal_proof_agent",
                }
            )
            informal_proof = extract_informal_proof(
                informal_proof_trace[0]["response"][0]
            )
            retrieval_traces = await db.find_traces(
                {
                    "state_id": result["statement_id"],
                    "agent_name": "theorem_retrieval_agent",
                }
            )
            retrieval_traces.sort(key=lambda trace: trace["created_at"], reverse=True)
            theorems = extract_theorems(retrieval_traces[0]["response"][0])
            data.append(
                {
                    "formal_statement": result["formal_statement"],
                    "retrieval_results": theorems,
                    "informal_proof": informal_proof,
                    "proof_sketch": sketch,
                    "success": result["success"],
                }
            )
    return data


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    file = args.file
    with open(file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]
    db = SQLiteClient(os.getenv("DATABASE_PATH"))
    retriever = Retriever()
    await db.connect()

    statement_formalization_data = await extract_statement_formalization(results, db)
    proof_generation_data = await extract_proof_generation_data(results, db)
    proof_correction_data = await extract_proof_correction_data(results, db)
    proof_sketching_data = await extract_proof_sketching_data(results, db)
    premise_selection_data = await extract_premise_selection_data(
        results, db, retriever
    )
    await db.disconnect()
    with open("results/statement_formalization_data.jsonl", "w") as f:
        for data in statement_formalization_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open("results/premise_selection_data.jsonl", "w") as f:
        for data in premise_selection_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open("results/proof_generation_data.jsonl", "w") as f:
        for data in proof_generation_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open("results/proof_correction_data.jsonl", "w") as f:
        for data in proof_correction_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open("results/proof_sketching_data.jsonl", "w") as f:
        for data in proof_sketching_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
