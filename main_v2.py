"""
Main entry point for TheoremForge V2 with full async support.

This demonstrates the new async manager that can:
- Continuously accept new requests
- Process multiple theorems concurrently
- Handle dynamic workload additions
"""

import asyncio
from datasets import load_dataset
import re
from theoremforge.manager import TheoremForgeStateManager
from loguru import logger


def erase_header(code: str) -> str:
    """Remove header statements from Lean code."""
    import_pattern = re.compile(r"^import\s+.*$", re.MULTILINE)
    open_pattern = re.compile(r"^open\s+.*$", re.MULTILINE)
    set_option_pattern = re.compile(r"^set_option\s+.*$", re.MULTILINE)
    return re.sub(
        import_pattern, "",
        re.sub(open_pattern, "", re.sub(set_option_pattern, "", code))
    )


async def main():
    """Main async function."""
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("minif2f_test")["train"][:100]
    statements = [erase_header(statement) for statement in dataset["formal_statement"]]
    logger.info(f"Loaded {len(statements)} formal statements")

    # Initialize manager with concurrent workers
    manager = TheoremForgeStateManager(
        max_workers=5,  # 5 concurrent workers per stage
        output_file="state_trace.jsonl"
    )

    # Start the manager
    await manager.start()

    try:
        # Submit all statements (they will be processed concurrently)
        logger.info("Submitting all statements...")
        await manager.submit_multiple(statements, batch_size=20)

        # Monitor progress
        while True:
            stats = manager.get_stats()
            finished = stats['total_finished']
            total = stats['total_submitted']

            if finished >= total:
                break

            # Log progress every few seconds
            logger.info(
                f"Progress: {finished}/{total} completed "
                f"(Active: {stats['active_tasks']}, "
                f"Success: {stats['successful']}, Failed: {stats['failed']})"
            )
            await asyncio.sleep(5)

        # Wait for final completion
        await manager.wait_for_completion()

    finally:
        # Graceful shutdown
        await manager.stop(timeout=120)


async def demo_continuous_submission():
    """
    Demonstrates continuous submission capability.

    This shows how new theorems can be added while others are still processing.
    """
    logger.info("=== Continuous Submission Demo ===")

    # Load dataset
    dataset = load_dataset("minif2f_test")["train"][:50]
    statements = [erase_header(statement) for statement in dataset["formal_statement"]]

    # Initialize manager
    manager = TheoremForgeStateManager(
        max_workers=10,
        output_file="state_trace_demo.jsonl"
    )

    await manager.start()

    try:
        # Submit statements in batches with delays
        batch_size = 10
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i+batch_size]
            logger.info(f"Submitting batch {i//batch_size + 1}...")
            await manager.submit_multiple(batch)

            # Log current stats
            stats = manager.get_stats()
            logger.info(
                f"Current stats - Submitted: {stats['total_submitted']}, "
                f"Finished: {stats['total_finished']}, Active: {stats['active_tasks']}"
            )

            # Small delay before next batch
            await asyncio.sleep(5)

        # Wait for all to complete
        await manager.wait_for_completion()

    finally:
        await manager.stop()


async def demo_dynamic_workload():
    """
    Demonstrates adding work dynamically based on results.

    This could be used for adaptive theorem proving strategies.
    """
    logger.info("=== Dynamic Workload Demo ===")

    # Simple example statements
    statements = [
        "theorem example1 : 1 + 1 = 2 := by sorry",
        "theorem example2 : 2 + 2 = 4 := by sorry",
        "theorem example3 : 3 + 3 = 6 := by sorry",
    ]

    manager = TheoremForgeStateManager(
        max_workers=2,
        output_file="state_trace_dynamic.jsonl"
    )

    await manager.start()

    try:
        # Submit initial batch
        logger.info("Submitting initial theorems...")
        await manager.submit_multiple(statements[:2])

        # Wait a bit
        await asyncio.sleep(10)

        # Check stats and add more work
        stats = manager.get_stats()
        logger.info(f"Mid-processing stats: {stats}")

        # Add more work dynamically
        logger.info("Adding more theorems dynamically...")
        await manager.submit_formal_statement(statements[2])

        # Wait for completion
        await manager.wait_for_completion()

    finally:
        await manager.stop()


if __name__ == "__main__":
    # Choose which demo to run
    import sys

    if len(sys.argv) > 1:
        demo = sys.argv[1]
        if demo == "continuous":
            asyncio.run(demo_continuous_submission())
        elif demo == "dynamic":
            asyncio.run(demo_dynamic_workload())
        else:
            logger.error(f"Unknown demo: {demo}")
            logger.info("Available demos: continuous, dynamic")
            sys.exit(1)
    else:
        # Run main by default
        asyncio.run(main())




