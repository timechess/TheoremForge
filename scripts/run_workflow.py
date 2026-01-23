from theoremforge.manager import TheoremForgeStateManager
import asyncio
import argparse
import json
import os
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def load_existing_results(export_file: str) -> tuple[list[dict], set]:
    """
    Load existing output results, find the last entry with success=true,
    keep it and all previous results, return the kept results list and the set of processed ids.
    """
    all_results = []
    
    if os.path.exists(export_file):
        with open(export_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_results.append(data)
                    except json.JSONDecodeError:
                        continue
    
    if not all_results:
        return [], set()
    
    # Find the index of the last entry with success=true
    last_success_idx = -1
    for i, result in enumerate(all_results):
        if result.get("success", False):
            last_success_idx = i
    
    if last_success_idx == -1:
        # No successful entries, start from the beginning
        logger.info("No successful entries found, will start processing from the beginning")
        return [], set()
    
    # Keep the last successful entry and all previous results
    kept_results = all_results[:last_success_idx + 1]
    processed_ids = {result["id"] for result in kept_results}
    
    discarded_count = len(all_results) - len(kept_results)
    logger.info(
        f"Detected {len(all_results)} existing results, "
        f"last successful is entry {last_success_idx + 1}, "
        f"keeping {len(kept_results)} entries, discarding {discarded_count} failed results"
    )
    
    return kept_results, processed_ids


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--export_file", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    # Load existing results
    existing_results, processed_ids = [], set()
    if args.resume:
        existing_results, processed_ids = load_existing_results(args.export_file)

    manager = TheoremForgeStateManager(
        max_workers=args.max_workers,
        config_path=args.config_path,
    )
    await manager.start()
    
    id_map = {}
    skipped_count = 0
    
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # Skip already processed entries
            if data["id"] in processed_ids:
                skipped_count += 1
                continue
            
            statment_id = await manager.submit_informal_statement(
                informal_statement=data["nl_problem"],
            )
            id_map[data["id"]] = statment_id
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} already processed entries, newly submitted {len(id_map)} entries")
    
    if not id_map:
        logger.info("No new entries to process")
        await manager.stop()
        return
    
    await manager.wait_for_completion()
    
    # Collect new results
    new_results = []
    for i, statement_id in id_map.items():
        data = await manager.context.db.get_state(statement_id)
        if data:
            new_results.append({
                "id": i,
                "statement_id": statement_id,
                "formal_statement": data.get("formal_statement", ""),
                "informal_statement": data.get("informal_statement", ""),
                "formal_proof": data.get("formal_proof", ""),
                "success": data.get("success", False),
            })
    
    # Merge existing results and new results, write to output file
    all_results = existing_results + new_results
    with open(args.export_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Processing completed, total {len(all_results)} results (existing {len(existing_results)} + new {len(new_results)})")
    
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
