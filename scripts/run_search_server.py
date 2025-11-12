from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uvicorn
from loguru import logger
import argparse

app = FastAPI(
    title="Lean Search Server", description="Semantic search for Lean constants"
)


# Initialize clients and model
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")


# Request/Response models
class SearchRequest(BaseModel):
    queries: List[str]
    topk: int = 5
    collection_name: str


class SearchResult(BaseModel):
    id: int
    score: float
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[List[SearchResult]]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar documents given a list of queries.
    Returns topk results for each query with their payloads.
    """
    try:
        # Embed queries using CPU
        query_embeddings = model.encode(
            request.queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_results = []
        # Search for each query (only theorems)
        for query_embedding in query_embeddings:
            search_results = client.query_points(
                collection_name=request.collection_name,
                query=query_embedding.tolist(),
                limit=request.topk,
                with_payload=[
                    "name",
                    "kind",
                    "type",
                    "informal_name",
                    "informal_description",
                ],
            )
            # Format results
            formatted_results = [
                SearchResult(
                    id=result.id, score=result.score, payload=result.payload or {}
                )
                for result in search_results.points
            ]

            all_results.append(formatted_results)
        logger.info(f"Search query: {request.queries}")
        return SearchResponse(results=all_results)

    except Exception as e:
        print(request)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "all-mpnet-base-v2", "device": "cpu"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on"
    )
    args = parser.parse_args()
    print("Starting Lean Search Server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
