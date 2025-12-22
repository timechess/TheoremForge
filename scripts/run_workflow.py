from theoremforge.manager import TheoremForgeStateManager
import asyncio
import argparse
import json


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--export_file", type=str, required=True)

    args = parser.parse_args()

    manager = TheoremForgeStateManager(
        max_workers=args.max_workers,
        config_path=args.config_path,
    )
    await manager.start()
    id_map = {}
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            statment_id = await manager.submit_informal_statement(
                informal_statement=data["nl_problem"],
            )
            id_map[data["id"]] = statment_id
    await manager.wait_for_completion()
    with open(args.export_file, "w") as f:
        for i, statement_id in id_map.items():
            data = await manager.context.db.get_state(statement_id)
            if data:
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "statement_id": statement_id,
                            "formal_statement": data.get("formal_statement", ""),
                            "informal_statement": data.get("informal_statement", ""),
                            "formal_proof": data.get("formal_proof", ""),
                            "success": data.get("success", False),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
