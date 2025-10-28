from qdrant_client import QdrantClient, models
import asyncio
import argparse
import tqdm
import torch
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset


def encode_on_gpu(args):
    """Function to run embedding on a specific GPU in a separate process."""
    texts_chunk, gpu_id, model_name, batch_size = args

    # Set the specific GPU for this process
    device_name = f"cuda:{gpu_id}"
    print(f"Process starting on {device_name} with {len(texts_chunk)} texts")

    # Load model directly on the specified GPU
    model = SentenceTransformer(model_name, device=device_name)

    # Generate embeddings
    embeddings = model.encode(
        texts_chunk,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device_name,
    )

    print(f"Process finished on {device_name}")
    return embeddings


def get_embeddings(
    texts,
    batch_size=32,
    use_multi_gpu=True,
    gpu_ids=None,
    model_name="all-mpnet-base-v2",
):
    """Generate embeddings using sentence-transformers with multi-GPU acceleration."""
    gpu_count = torch.cuda.device_count()
    if use_multi_gpu and gpu_count > 1:
        # Determine which GPUs to use
        if gpu_ids:
            available_gpus = gpu_ids
            print(f"Using multi-GPU encoding with specified GPUs: {gpu_ids}")
        else:
            available_gpus = list(range(gpu_count))
            print(f"Using multi-GPU encoding with all {gpu_count} GPUs")

        num_processes = len(available_gpus)

        # Split texts into chunks for each GPU
        chunk_size = max(1, len(texts) // num_processes)
        text_chunks = []

        for i in range(num_processes):
            start_idx = i * chunk_size
            if i == num_processes - 1:  # Last chunk gets remaining texts
                end_idx = len(texts)
            else:
                end_idx = (i + 1) * chunk_size

            if start_idx < len(texts):  # Only add non-empty chunks
                text_chunks.append(texts[start_idx:end_idx])

        # Filter out GPUs for empty chunks
        actual_gpus = available_gpus[: len(text_chunks)]
        num_processes = len(text_chunks)

        print(
            f"Split {len(texts)} texts into {num_processes} chunks: {[len(chunk) for chunk in text_chunks]}"
        )

        # Prepare arguments for each process
        process_args = [
            (chunk, gpu_id, model_name, batch_size)
            for chunk, gpu_id in zip(text_chunks, actual_gpus)
        ]
        # Use ProcessPoolExecutor to run on multiple GPUs
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(encode_on_gpu, process_args),
                    total=num_processes,
                    desc="Processing on GPUs",
                )
            )

        # Concatenate results
        embeddings = np.concatenate(results, axis=0)
        print(f"Generated {len(embeddings)} embeddings")

        return embeddings
    else:
        print("Using single GPU/CPU encoding")
        model = SentenceTransformer(model_name)
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--no_multi_gpu", action="store_true", help="Disable multi-GPU processing"
    )
    parser.add_argument(
        "--gpu_ids", type=str, help="Comma-separated GPU IDs to use (e.g., '0,1,2')"
    )
    args = parser.parse_args()
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")

    # Model configuration
    model_name = "all-mpnet-base-v2"

    # Check GPU availability and configuration
    num_gpus = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Available GPUs: {num_gpus}")
    print(f"Using device: {device}")

    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    # Load sentence transformer model
    model = SentenceTransformer(model_name)
    # Create collection with fixed vector size
    print(
        f"Creating collection with fixed vector size {model.get_sentence_embedding_dimension()}"
    )
    if client.collection_exists("lean-search-server"):
        client.delete_collection("lean-search-server")

    client.create_collection(
        collection_name="lean-search-server",
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    del model

    texts = []
    metadata = []

    print("Loading documents...")
    dataset = load_dataset(args.dataset_name)
    for doc in dataset["train"]:
        texts.append(doc["informal_description"])
        metadata.append(
            {
                "name": ".".join(doc["name"]),
                "type": doc["type"],
                "informal_name": doc["informal_name"],
                "informal_description": doc["informal_description"],
            }
        )

    # Parse GPU IDs if specified
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        print(f"Using specified GPU IDs: {gpu_ids}")

    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = get_embeddings(
        texts,
        batch_size=args.batch_size,
        use_multi_gpu=not args.no_multi_gpu,
        gpu_ids=gpu_ids,
        model_name=model_name,
    )

    print("Uploading vectors to Qdrant...")
    points = []
    for i, (embedding, payload) in enumerate(zip(embeddings, metadata)):
        points.append(
            models.PointStruct(id=i, vector=embedding.tolist(), payload=payload)
        )

    # Upload in batches to avoid memory issues
    batch_size = 100
    for i in tqdm.tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
        batch_points = points[i : i + batch_size]
        client.upsert(collection_name="lean-search-server", points=batch_points)


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
