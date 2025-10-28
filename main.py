from theoremforge.manager import run_theorem_forge
import asyncio
from datasets import load_dataset
from theoremforge.lean_server.server import erase_header

dataset = load_dataset("AI-MO/minif2f_test")["train"]

statements = [erase_header(statement["formal_statement"]) for statement in dataset]

async def main():
    await run_theorem_forge(statements, max_workers=10)


if __name__ == "__main__":
    asyncio.run(main())
