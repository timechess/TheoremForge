from theoremforge.manager import run_theorem_forge
import asyncio
import argparse
import json
import os
from theoremforge.utils import remove_comments

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--export_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--specific_ids", type=str, required=False)
parser.add_argument(
    "--retry_failed",
    action="store_true",
    help="Only retry problems that are not in the output file",
)
args = parser.parse_args()

# Get the list of completed IDs if retrying failed problems
completed_ids = set()
if args.retry_failed:
    output_path = "competition_result/" + args.output_file
    if os.path.exists(output_path):
        print(f"Retry mode: Loading completed IDs from {output_path}")
        with open(output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Only consider it completed if it has a formal_statement
                    if data.get("formal_code"):
                        completed_ids.add(data["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {len(completed_ids)} completed problems")
    else:
        print(
            f"Retry mode: Output file {output_path} does not exist, will process all problems"
        )

# Load problems from dataset
if not args.specific_ids:
    informal_statements = []
    ids = []
    with open(args.dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Skip if in retry mode and already completed
            if args.retry_failed and data["id"] in completed_ids:
                continue
            informal_statements.append(data["nl_problem"])
            ids.append(data["id"])
else:
    specific_ids = args.specific_ids.split(",")
    informal_statements = []
    ids = []
    with open(args.dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] in specific_ids:
                # Skip if in retry mode and already completed
                if args.retry_failed and data["id"] in completed_ids:
                    continue
                informal_statements.append(data["nl_problem"])
                ids.append(data["id"])

if args.retry_failed:
    print(f"Retry mode: Will process {len(ids)} failed/missing problems")
else:
    print(f"Will process {len(ids)} problems")


async def main():
    if not ids:
        print("No problems to process. Exiting.")
        return

    await run_theorem_forge(
        informal_statements=informal_statements,
        max_workers=2,
        custom_ids=ids,
        export_file=args.export_file,
    )

    # Process the export file and write to output
    output_path = "competition_result/" + args.output_file

    # In retry mode, append to existing file; otherwise, overwrite
    mode = "a" if args.retry_failed and os.path.exists(output_path) else "w"

    with open(args.export_file, "r") as f:
        with open(output_path, mode) as g:
            for line in f:
                data = json.loads(line)
                if not data["formal_proof"]:
                    continue
                formal_statement = (
                    data.get("formal_statement", "").rsplit(":=", 1)[0] + ":="
                    if ":=" in data.get("formal_statement", "")
                    else data.get("formal_statement", "")
                )
                record = {
                    "id": data["custom_id"],
                    "nl_problem": data["informal_statement"],
                    "formal_type": "Lean",
                    "header": data["header"]
                    + data.get("formal_proof", "").split(formal_statement)[0],
                    "formal_statement": formal_statement,
                    "formal_code": data["header"] + data.get("formal_proof", ""),
                }
                g.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.retry_failed:
        print(f"Retry mode: Results appended to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
