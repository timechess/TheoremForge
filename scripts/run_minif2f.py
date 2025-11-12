from datasets import load_dataset
from theoremforge.manager import run_theorem_forge
from theoremforge.lean_server.server import erase_header
import asyncio
import argparse

minif2f = load_dataset("AI-MO/minif2f_test")["train"]

formal_statements = [erase_header(item["formal_statement"]) for item in minif2f]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    args = parser.parse_args()

    await run_theorem_forge(
        formal_statements=formal_statements,
        max_workers=int(args.max_workers),
        export_file=args.output_file,
    )


if __name__ == "__main__":
    asyncio.run(main())
