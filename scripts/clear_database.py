"""
Clear MongoDB Database

This script removes all data from the TheoremForge MongoDB database.
Use with caution - this operation cannot be undone!
"""

import asyncio
import sys
from theoremforge.db import MongoDBClient
from loguru import logger


async def clear_database(drop_database: bool = False):
    """
    Clear all collections or drop the entire database.

    Args:
        drop_database: If True, drop the entire database.
                      If False, just clear all collections.
    """
    logger.info("=" * 60)
    logger.info("MongoDB Database Clear Utility")
    logger.info("=" * 60)

    # Connect to database
    db = MongoDBClient()

    try:
        await db.connect()
        logger.info("✓ Connected to MongoDB")

        if drop_database:
            # Drop the entire database
            logger.warning("Dropping entire database...")
            await db.client.drop_database(db.db.name)
            logger.info("✓ Database dropped successfully")
        else:
            # Clear all collections
            collections = [
                ("theorem_forge_states", "TheoremForge States"),
                ("prover_traces", "Prover Traces"),
                ("self_correction_traces", "Self Correction Traces"),
                ("theorem_retrieval_traces", "Theorem Retrieval Traces"),
                ("informal_proof_traces", "Informal Proof Traces"),
                ("proof_sketch_traces", "Proof Sketch Traces"),
                ("proof_assembly_traces", "Proof Assembly Traces"),
            ]

            logger.info(f"Clearing {len(collections)} collections...")

            for collection_name, display_name in collections:
                collection = db.db[collection_name]

                # Count before deletion
                count_before = await collection.count_documents({})

                if count_before > 0:
                    # Delete all documents
                    result = await collection.delete_many({})
                    logger.info(f"✓ {display_name}: Deleted {result.deleted_count} documents")
                else:
                    logger.info(f"  {display_name}: Already empty")

        logger.info("=" * 60)
        logger.info("Database cleared successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Failed to clear database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await db.disconnect()


async def show_stats():
    """Show current database statistics."""
    logger.info("=" * 60)
    logger.info("Current Database Statistics")
    logger.info("=" * 60)

    db = MongoDBClient()

    try:
        await db.connect()

        collections = [
            ("theorem_forge_states", "TheoremForge States"),
            ("prover_traces", "Prover Traces"),
            ("self_correction_traces", "Self Correction Traces"),
            ("theorem_retrieval_traces", "Theorem Retrieval Traces"),
            ("informal_proof_traces", "Informal Proof Traces"),
            ("proof_sketch_traces", "Proof Sketch Traces"),
            ("proof_assembly_traces", "Proof Assembly Traces"),
        ]

        total = 0
        for collection_name, display_name in collections:
            collection = db.db[collection_name]
            count = await collection.count_documents({})
            total += count
            logger.info(f"  {display_name}: {count} documents")

        logger.info("=" * 60)
        logger.info(f"Total: {total} documents")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Failed to get stats: {e}")
        sys.exit(1)
    finally:
        await db.disconnect()


async def interactive_clear():
    """Interactive mode with confirmation."""
    print("\n" + "=" * 60)
    print("TheoremForge MongoDB Database Clear Utility")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Show database statistics")
    print("  2. Clear all collections (keep database structure)")
    print("  3. Drop entire database (complete reset)")
    print("  4. Exit")
    print()

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        await show_stats()

    elif choice == "2":
        await show_stats()
        print("\n⚠️  WARNING: This will delete ALL data from all collections!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()

        if confirm == "yes":
            await clear_database(drop_database=False)
        else:
            print("Operation cancelled.")

    elif choice == "3":
        await show_stats()
        print("\n⚠️  WARNING: This will DROP the entire database!")
        print("This is a more complete reset than clearing collections.")
        confirm = input("Type 'DELETE' (in capitals) to confirm: ").strip()

        if confirm == "DELETE":
            await clear_database(drop_database=True)
        else:
            print("Operation cancelled.")

    elif choice == "4":
        print("Exiting...")

    else:
        print("Invalid choice. Exiting...")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "--stats":
            asyncio.run(show_stats())
        elif sys.argv[1] == "--clear":
            print("⚠️  WARNING: This will delete ALL data!")
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                asyncio.run(clear_database(drop_database=False))
            else:
                print("Operation cancelled.")
        elif sys.argv[1] == "--drop":
            print("⚠️  WARNING: This will DROP the entire database!")
            confirm = input("Type 'DELETE' to confirm: ").strip()
            if confirm == "DELETE":
                asyncio.run(clear_database(drop_database=True))
            else:
                print("Operation cancelled.")
        elif sys.argv[1] == "--force-clear":
            # No confirmation (use with caution!)
            asyncio.run(clear_database(drop_database=False))
        elif sys.argv[1] == "--force-drop":
            # No confirmation (use with caution!)
            asyncio.run(clear_database(drop_database=True))
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python clear_database.py              # Interactive mode")
            print("  python clear_database.py --stats      # Show database statistics")
            print("  python clear_database.py --clear      # Clear all collections (with confirmation)")
            print("  python clear_database.py --drop       # Drop entire database (with confirmation)")
            print("  python clear_database.py --force-clear # Clear without confirmation")
            print("  python clear_database.py --force-drop  # Drop without confirmation")
            print("  python clear_database.py --help       # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Interactive mode
        asyncio.run(interactive_clear())


if __name__ == "__main__":
    main()

