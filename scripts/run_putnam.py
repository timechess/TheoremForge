import json
from theoremforge.manager import TheoremForgeStateManager
import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    manager = TheoremForgeStateManager(
        max_workers=args.max_workers,
        config_path=args.config_path,
    )
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]
    await manager.start()

    statement_ids = await manager.submit_multiple(formal_statements=[item["lean4_statement"] for item in data])
    await manager.wait_for_completion()
    await manager.export_results(args.output_file, statement_ids, [item["name"] for item in data])
    await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())