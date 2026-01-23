import argparse
import json
import random
from collections import defaultdict
from datasets import load_dataset


def normalize_deeptheorem_example(example):
    return {
        "id": f"deeptheorem_{example.get('id', '')}",
        "nl_problem": example.get("informal_theorem_qa", ""),
        "domain": example.get("domain", []),
        "difficulty": example.get("difficulty"),
    }


def normalize_deepmath_example(example, index):
    question = example.get("question", "")
    final_answer = example.get("final_answer", "")
    nl_problem = question
    if final_answer:
        nl_problem = f"{question}\n\nAnswer: {final_answer}"
    
    topic = example.get("topic", "")
    domain = [topic] if isinstance(topic, str) and topic else []
    if isinstance(topic, list):
        domain = topic
    
    generated_id = f"deepmath_{index}"
    
    return {
        "id": generated_id,
        "nl_problem": nl_problem,
        "domain": domain,
        "difficulty": example.get("difficulty"),
    }


def filter_by_difficulty_range(examples, min_difficulty, max_difficulty):
    filtered = []
    for example in examples:
        difficulty = example.get("difficulty")
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
    random.seed(seed)
    
    total_examples = len(dataset_examples)
    if total_examples == 0:
        return []
    
    if num_samples > total_examples:
        raise ValueError(f"Requested {num_samples} samples, but dataset only has {total_examples} samples")
    
    # Group by difficulty (convert float difficulty values to strings as keys)
    difficulty_groups = defaultdict(list)
    for idx, example in enumerate(dataset_examples):
        difficulty = example.get("difficulty")
        # Skip if difficulty is None
        if difficulty is None:
            continue
        # Convert float to string as grouping key
        difficulty_key = str(difficulty)
        difficulty_groups[difficulty_key].append(idx)
    
    # Calculate original distribution for each difficulty level
    difficulty_proportions = {
        diff: len(indices) / total_examples 
        for diff, indices in difficulty_groups.items()
    }
    
    print(f"Total dataset size: {total_examples}")
    print("Difficulty distribution:")
    for diff, prop in sorted(difficulty_proportions.items()):
        count = len(difficulty_groups[diff])
        print(f"  {diff}: {count} ({prop*100:.2f}%)")
    
    # Calculate sampling count for each difficulty level based on original distribution (use float accumulation for precision)
    sampled_indices = []
    difficulty_samples = {}
    total_allocated = 0
    
    # First round: allocate proportionally (round down)
    for difficulty, indices in sorted(difficulty_groups.items()):
        proportion = difficulty_proportions[difficulty]
        num_samples_for_diff = int(num_samples * proportion)
        num_samples_for_diff = min(num_samples_for_diff, len(indices))
        difficulty_samples[difficulty] = num_samples_for_diff
        total_allocated += num_samples_for_diff
    
    # Second round: allocate remaining samples (by proportion priority)
    remaining = num_samples - total_allocated
    if remaining > 0:
        # Sort by proportion, prioritize higher proportion difficulty levels
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
    
    # Sample from each difficulty level
    for difficulty, num_samples_for_diff in difficulty_samples.items():
        if num_samples_for_diff > 0:
            indices = difficulty_groups[difficulty]
            sampled = random.sample(indices, num_samples_for_diff)
            sampled_indices.extend(sampled)
            print(f"Sampled {num_samples_for_diff} samples from difficulty {difficulty}")
    
    # If still insufficient (should not happen in theory), randomly supplement
    if len(sampled_indices) < num_samples:
        remaining = num_samples - len(sampled_indices)
        all_indices = set(range(total_examples))
        remaining_indices = list(all_indices - set(sampled_indices))
        if remaining_indices:
            additional_samples = random.sample(remaining_indices, min(remaining, len(remaining_indices)))
            sampled_indices.extend(additional_samples)
            print(f"Supplemental sampling: {len(additional_samples)} samples")
    
    # If exceeds (should not happen in theory), randomly reduce
    if len(sampled_indices) > num_samples:
        sampled_indices = random.sample(sampled_indices, num_samples)
        print(f"Reduced sampling: keeping {num_samples} samples")
    
    # Shuffle order
    random.shuffle(sampled_indices)
    
    # Select corresponding samples
    sampled_examples = [dataset_examples[idx] for idx in sampled_indices]
    
    print(f"\nFinal sampling count: {len(sampled_examples)}")
    
    # Strict check
    assert len(sampled_examples) == num_samples, f"Sampling count mismatch: expected {num_samples}, got {len(sampled_examples)}"
    
    return sampled_examples


def main():
    parser = argparse.ArgumentParser(description="Sample data from DeepTheorem and DeepMath datasets by difficulty distribution")
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Total number of data points to sample"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        required=True,
        help="Dataset ratio, format: deeptheorem:deepmath, e.g., 0.6:0.4"
    )
    parser.add_argument(
        "--min_difficulty",
        type=float,
        default=float('-inf'),
        help="Minimum difficulty value (float type), data below this difficulty will be excluded (default: no limit)"
    )
    parser.add_argument(
        "--max_difficulty",
        type=float,
        default=float('inf'),
        help="Maximum difficulty value (float type), data above this difficulty will be excluded (default: no limit)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, if not specified, only print the first sample)"
    )
    
    args = parser.parse_args()
    
    # Parse ratio
    ratio_parts = args.ratio.split(":")
    if len(ratio_parts) != 2:
        raise ValueError("Ratio format error, should be deeptheorem:deepmath, e.g., 0.6:0.4")
    
    try:
        ratio_deeptheorem = float(ratio_parts[0])
        ratio_deepmath = float(ratio_parts[1])
    except ValueError:
        raise ValueError("Ratio must be numeric")
    
    # Normalize ratio
    total_ratio = ratio_deeptheorem + ratio_deepmath
    ratio_deeptheorem /= total_ratio
    ratio_deepmath /= total_ratio
    
    num_deeptheorem = int(args.num_samples * ratio_deeptheorem)
    num_deepmath = args.num_samples - num_deeptheorem
    
    print("Sampling configuration:")
    print(f"  Total samples: {args.num_samples}")
    print(f"  DeepTheorem: {num_deeptheorem} ({ratio_deeptheorem*100:.1f}%)")
    print(f"  DeepMath: {num_deepmath} ({ratio_deepmath*100:.1f}%)")
    min_diff_str = "no limit" if args.min_difficulty == float('-inf') else str(args.min_difficulty)
    max_diff_str = "no limit" if args.max_difficulty == float('inf') else str(args.max_difficulty)
    print(f"  Difficulty range: [{min_diff_str}, {max_diff_str}]")
    print(f"  Random seed: {args.seed}")
    
    # Load datasets
    print("\nLoading DeepTheorem dataset...")
    deeptheorem_raw = load_dataset("Jiahao004/DeepTheorem")["train"]
    
    print("Loading DeepMath dataset...")
    deepmath_raw = load_dataset("zwhe99/DeepMath-103K")["train"]
    
    # Normalize data
    print("\nNormalizing DeepTheorem data...")
    deeptheorem_examples = [normalize_deeptheorem_example(ex) for ex in deeptheorem_raw]
    
    print("Normalizing DeepMath data...")
    deepmath_examples = [normalize_deepmath_example(ex, idx) for idx, ex in enumerate(deepmath_raw)]
    
    # Filter by difficulty range
    min_diff_str = "no limit" if args.min_difficulty == float('-inf') else str(args.min_difficulty)
    max_diff_str = "no limit" if args.max_difficulty == float('inf') else str(args.max_difficulty)
    print(f"\nFiltering data by difficulty range [{min_diff_str}, {max_diff_str}]...")
    deeptheorem_filtered = filter_by_difficulty_range(deeptheorem_examples, args.min_difficulty, args.max_difficulty)
    deepmath_filtered = filter_by_difficulty_range(deepmath_examples, args.min_difficulty, args.max_difficulty)
    
    print(f"DeepTheorem after filtering: {len(deeptheorem_filtered)} / {len(deeptheorem_examples)}")
    print(f"DeepMath after filtering: {len(deepmath_filtered)} / {len(deepmath_examples)}")
    
    # Sample from both datasets separately
    print(f"\nSampling {num_deeptheorem} samples from DeepTheorem...")
    sampled_deeptheorem = sample_by_difficulty(
        deeptheorem_filtered,
        num_deeptheorem,
        seed=args.seed
    )
    
    print(f"\nSampling {num_deepmath} samples from DeepMath...")
    sampled_deepmath = sample_by_difficulty(
        deepmath_filtered,
        num_deepmath,
        seed=args.seed
    )
    
    # Merge results
    all_sampled = sampled_deeptheorem + sampled_deepmath
    random.seed(args.seed)
    random.shuffle(all_sampled)
    
    print(f"\nTotal samples after merging: {len(all_sampled)}")
    
    # Strict check on total sample count
    assert len(all_sampled) == args.num_samples, f"Total sampling count mismatch: expected {args.num_samples}, got {len(all_sampled)}"
    
    # Save or display results
    if args.output:
        print(f"Saving sampling results to {args.output}...")
        fields_to_keep = ["id", "nl_problem", "domain", "difficulty"]
        with open(args.output, "w", encoding="utf-8") as f:
            for example in all_sampled:
                filtered_example = {
                    field: example.get(field) 
                    for field in fields_to_keep 
                    if field in example
                }
                f.write(json.dumps(filtered_example, ensure_ascii=False) + "\n")
        print("Save completed!")
    else:
        print("\nFirst sampled sample:")
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
