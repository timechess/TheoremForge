import argparse
import json
import random
from collections import defaultdict
from datasets import load_dataset


def normalize_deeptheorem_example(example):
    """标准化 DeepTheorem 数据集的样本"""
    return {
        "id": f"deeptheorem_{example.get('id', '')}",
        "nl_problem": example.get("informal_theorem_qa", ""),
        "domain": example.get("domain", []),
        "difficulty": example.get("difficulty"),
    }


def normalize_deepmath_example(example, index):
    """标准化 DeepMath 数据集的样本：合并 question 和 final_answer，将 topic 转为列表，生成 id"""
    question = example.get("question", "")
    final_answer = example.get("final_answer", "")
    # 合并 question 和 final_answer
    nl_problem = question
    if final_answer:
        nl_problem = f"{question}\n\nAnswer: {final_answer}"
    
    topic = example.get("topic", "")
    # 将 topic 转为列表
    domain = [topic] if isinstance(topic, str) and topic else []
    if isinstance(topic, list):
        domain = topic
    
    # 为 DeepMath 生成 id（格式：deepmath_<index>）
    generated_id = f"deepmath_{index}"
    
    return {
        "id": generated_id,
        "nl_problem": nl_problem,
        "domain": domain,
        "difficulty": example.get("difficulty"),
    }


def filter_by_difficulty_range(examples, min_difficulty, max_difficulty):
    """根据难度范围过滤样本（difficulty 为 float 类型）"""
    filtered = []
    for example in examples:
        difficulty = example.get("difficulty")
        # 如果 difficulty 为 None 或不是数字，跳过
        if difficulty is None:
            continue
        try:
            difficulty_float = float(difficulty)
            if min_difficulty <= difficulty_float <= max_difficulty:
                filtered.append(example)
        except (ValueError, TypeError):
            continue
    return filtered


def sample_by_difficulty(dataset_examples, num_samples, seed=42):
    """
    按照难度分布对齐采样数据，严格保证采样数量与用户输入一致
    
    Args:
        dataset_examples: 样本列表（已标准化的字典列表）
        num_samples: 需要采样的总数量
        seed: 随机种子
    
    Returns:
        采样后的样本列表（长度严格等于 num_samples）
    """
    # 设置随机种子
    random.seed(seed)
    
    total_examples = len(dataset_examples)
    if total_examples == 0:
        return []
    
    if num_samples > total_examples:
        raise ValueError(f"请求采样 {num_samples} 个样本，但数据集只有 {total_examples} 个样本")
    
    # 按难度分组（将 float 难度值转换为字符串作为键）
    difficulty_groups = defaultdict(list)
    for idx, example in enumerate(dataset_examples):
        difficulty = example.get("difficulty")
        # 如果 difficulty 为 None，跳过
        if difficulty is None:
            continue
        # 将 float 转换为字符串作为分组键
        difficulty_key = str(difficulty)
        difficulty_groups[difficulty_key].append(idx)
    
    # 计算每个难度级别的原始分布
    difficulty_proportions = {
        diff: len(indices) / total_examples 
        for diff, indices in difficulty_groups.items()
    }
    
    print(f"数据集总大小: {total_examples}")
    print("难度分布:")
    for diff, prop in sorted(difficulty_proportions.items()):
        count = len(difficulty_groups[diff])
        print(f"  {diff}: {count} ({prop*100:.2f}%)")
    
    # 按照原始分布计算每个难度级别的采样数量（使用浮点数累加确保精确）
    sampled_indices = []
    difficulty_samples = {}
    total_allocated = 0
    
    # 第一轮：按比例分配（向下取整）
    for difficulty, indices in sorted(difficulty_groups.items()):
        proportion = difficulty_proportions[difficulty]
        num_samples_for_diff = int(num_samples * proportion)
        num_samples_for_diff = min(num_samples_for_diff, len(indices))
        difficulty_samples[difficulty] = num_samples_for_diff
        total_allocated += num_samples_for_diff
    
    # 第二轮：分配剩余样本（按比例优先级）
    remaining = num_samples - total_allocated
    if remaining > 0:
        # 按比例排序，优先分配给比例高的难度级别
        sorted_difficulties = sorted(
            difficulty_groups.items(),
            key=lambda x: difficulty_proportions[x[0]],
            reverse=True
        )
        for difficulty, indices in sorted_difficulties:
            if remaining <= 0:
                break
            current_count = difficulty_samples[difficulty]
            if current_count < len(indices):
                add_count = min(remaining, len(indices) - current_count)
                difficulty_samples[difficulty] += add_count
                remaining -= add_count
    
    # 从每个难度级别采样
    for difficulty, num_samples_for_diff in difficulty_samples.items():
        if num_samples_for_diff > 0:
            indices = difficulty_groups[difficulty]
            sampled = random.sample(indices, num_samples_for_diff)
            sampled_indices.extend(sampled)
            print(f"从 {difficulty} 级别采样: {num_samples_for_diff} 个样本")
    
    # 如果仍然不足（理论上不应该发生），随机补充
    if len(sampled_indices) < num_samples:
        remaining = num_samples - len(sampled_indices)
        all_indices = set(range(total_examples))
        remaining_indices = list(all_indices - set(sampled_indices))
        if remaining_indices:
            additional_samples = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
            sampled_indices.extend(additional_samples)
            print(f"补充采样: {len(additional_samples)} 个样本")
    
    # 如果超过（理论上不应该发生），随机减少
    if len(sampled_indices) > num_samples:
        sampled_indices = random.sample(sampled_indices, num_samples)
        print(f"减少采样: 保留 {num_samples} 个样本")
    
    # 打乱顺序
    random.shuffle(sampled_indices)
    
    # 选择对应的样本
    sampled_examples = [dataset_examples[idx] for idx in sampled_indices]
    
    print(f"\n最终采样数量: {len(sampled_examples)}")
    
    # 严格检查
    assert len(sampled_examples) == num_samples, f"采样数量不匹配: 期望 {num_samples}, 实际 {len(sampled_examples)}"
    
    return sampled_examples


def main():
    parser = argparse.ArgumentParser(description="从 DeepTheorem 和 DeepMath 数据集中按难度分布采样数据")
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="需要采样的总数据点数量"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        required=True,
        help="数据集比例，格式：deeptheorem:deepmath，例如 0.6:0.4"
    )
    parser.add_argument(
        "--min_difficulty",
        type=float,
        default=float('-inf'),
        help="最小难度值（float 类型），低于此难度的数据将被排除 (默认: 无限制)"
    )
    parser.add_argument(
        "--max_difficulty",
        type=float,
        default=float('inf'),
        help="最大难度值（float 类型），超过此难度的数据将被排除 (默认: 无限制)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (可选，如果不指定则只打印第一个样本)"
    )
    
    args = parser.parse_args()
    
    # 解析比例
    ratio_parts = args.ratio.split(":")
    if len(ratio_parts) != 2:
        raise ValueError("比例格式错误，应为 deeptheorem:deepmath，例如 0.6:0.4")
    
    try:
        ratio_deeptheorem = float(ratio_parts[0])
        ratio_deepmath = float(ratio_parts[1])
    except ValueError:
        raise ValueError("比例必须是数字")
    
    # 归一化比例
    total_ratio = ratio_deeptheorem + ratio_deepmath
    ratio_deeptheorem /= total_ratio
    ratio_deepmath /= total_ratio
    
    num_deeptheorem = int(args.num_samples * ratio_deeptheorem)
    num_deepmath = args.num_samples - num_deeptheorem
    
    print("采样配置:")
    print(f"  总采样数: {args.num_samples}")
    print(f"  DeepTheorem: {num_deeptheorem} ({ratio_deeptheorem*100:.1f}%)")
    print(f"  DeepMath: {num_deepmath} ({ratio_deepmath*100:.1f}%)")
    min_diff_str = "无限制" if args.min_difficulty == float('-inf') else str(args.min_difficulty)
    max_diff_str = "无限制" if args.max_difficulty == float('inf') else str(args.max_difficulty)
    print(f"  难度范围: [{min_diff_str}, {max_diff_str}]")
    print(f"  随机种子: {args.seed}")
    
    # 加载数据集
    print("\n加载 DeepTheorem 数据集...")
    deeptheorem_raw = load_dataset("Jiahao004/DeepTheorem")["train"]
    
    print("加载 DeepMath 数据集...")
    deepmath_raw = load_dataset("zwhe99/DeepMath-103K")["train"]
    
    # 标准化数据
    print("\n标准化 DeepTheorem 数据...")
    deeptheorem_examples = [normalize_deeptheorem_example(ex) for ex in deeptheorem_raw]
    
    print("标准化 DeepMath 数据...")
    deepmath_examples = [normalize_deepmath_example(ex, idx) for idx, ex in enumerate(deepmath_raw)]
    
    # 根据难度范围过滤
    min_diff_str = "无限制" if args.min_difficulty == float('-inf') else str(args.min_difficulty)
    max_diff_str = "无限制" if args.max_difficulty == float('inf') else str(args.max_difficulty)
    print(f"\n根据难度范围 [{min_diff_str}, {max_diff_str}] 过滤数据...")
    deeptheorem_filtered = filter_by_difficulty_range(deeptheorem_examples, args.min_difficulty, args.max_difficulty)
    deepmath_filtered = filter_by_difficulty_range(deepmath_examples, args.min_difficulty, args.max_difficulty)
    
    print(f"DeepTheorem 过滤后: {len(deeptheorem_filtered)} / {len(deeptheorem_examples)}")
    print(f"DeepMath 过滤后: {len(deepmath_filtered)} / {len(deepmath_examples)}")
    
    # 从两个数据集分别采样
    print(f"\n从 DeepTheorem 采样 {num_deeptheorem} 个样本...")
    sampled_deeptheorem = sample_by_difficulty(
        deeptheorem_filtered,
        num_deeptheorem,
        seed=args.seed
    )
    
    print(f"\n从 DeepMath 采样 {num_deepmath} 个样本...")
    sampled_deepmath = sample_by_difficulty(
        deepmath_filtered,
        num_deepmath,
        seed=args.seed
    )
    
    # 合并结果
    all_sampled = sampled_deeptheorem + sampled_deepmath
    random.seed(args.seed)
    random.shuffle(all_sampled)
    
    print(f"\n合并后总样本数: {len(all_sampled)}")
    
    # 严格检查总样本数
    assert len(all_sampled) == args.num_samples, f"总采样数量不匹配: 期望 {args.num_samples}, 实际 {len(all_sampled)}"
    
    # 保存或显示结果
    if args.output:
        print(f"保存采样结果到 {args.output}...")
        fields_to_keep = ["id", "nl_problem", "domain", "difficulty"]
        with open(args.output, "w", encoding="utf-8") as f:
            for example in all_sampled:
                filtered_example = {
                    field: example.get(field) 
                    for field in fields_to_keep 
                    if field in example
                }
                f.write(json.dumps(filtered_example, ensure_ascii=False) + "\n")
        print("保存完成!")
    else:
        print("\n第一个采样样本:")
        example = all_sampled[0]
        fields_to_show = ["id", "nl_problem", "domain", "difficulty"]
        filtered_example = {
            field: example.get(field) 
            for field in fields_to_show 
            if field in example
        }
        print(json.dumps(filtered_example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
