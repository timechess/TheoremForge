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
    加载已有的输出结果，找到最后一个 success=true 的条目，
    保留它及之前的所有结果，返回保留的结果列表和已处理的 id 集合。
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
    
    # 找到最后一个 success=true 的条目索引
    last_success_idx = -1
    for i, result in enumerate(all_results):
        if result.get("success", False):
            last_success_idx = i
    
    if last_success_idx == -1:
        # 没有成功的条目，从头开始
        logger.info("没有找到成功的条目，将从头开始处理")
        return [], set()
    
    # 保留最后一个成功条目及之前的所有结果
    kept_results = all_results[:last_success_idx + 1]
    processed_ids = {result["id"] for result in kept_results}
    
    discarded_count = len(all_results) - len(kept_results)
    logger.info(
        f"检测到已有 {len(all_results)} 条结果，"
        f"最后成功的是第 {last_success_idx + 1} 条，"
        f"保留 {len(kept_results)} 条，丢弃 {discarded_count} 条失败结果"
    )
    
    return kept_results, processed_ids


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--export_file", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="从断点继续运行")

    args = parser.parse_args()

    # 加载已有结果
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
            # 跳过已处理的条目
            if data["id"] in processed_ids:
                skipped_count += 1
                continue
            
            statment_id = await manager.submit_informal_statement(
                informal_statement=data["nl_problem"],
            )
            id_map[data["id"]] = statment_id
    
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 条已处理的条目，新提交 {len(id_map)} 条")
    
    if not id_map:
        logger.info("没有新的条目需要处理")
        await manager.stop()
        return
    
    await manager.wait_for_completion()
    
    # 收集新结果
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
    
    # 合并已有结果和新结果，写入输出文件
    all_results = existing_results + new_results
    with open(args.export_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"处理完成，共 {len(all_results)} 条结果（已有 {len(existing_results)} 条 + 新增 {len(new_results)} 条）")
    
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
