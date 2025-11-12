from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from theoremforge.config import config
from theoremforge.utils import get_error_str
from theoremforge.lean_server.server import AsyncVerifier, erase_header

lean_config = config.lean_server


class VerificationRequest(BaseModel):
    code: str
    allow_sorry: bool = True


class VerificationResponse(BaseModel):
    valid: bool
    messages: List[dict]
    error_str: str


class ExtractSubgoalsRequest(BaseModel):
    code: str


class ExtractSubgoalsResponse(BaseModel):
    subgoals: List[str]


HEADER = lean_config.get("LeanServerHeader", "import Mathlib\n")

lean_config = config.lean_server
local_project = lean_config.get("LocalLeanProject")
workers = lean_config.get("LeanServerWorkers", 8)

if local_project:
    verifier = AsyncVerifier(project=local_project, workers=workers)
else:
    raise ValueError("No lean project specified")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    verifier.initialize_worker_pool()
    yield
    # Shutdown
    verifier.shutdown()


app = FastAPI(title="Lean Verifier API", lifespan=lifespan)


@app.post("/verify", response_model=VerificationResponse)
async def verify_code(request: VerificationRequest):
    """
    Verify a single piece of Lean code.

    Args:
        request: The verification request containing the code and allow_sorry flag

    Returns:
        VerificationResponse containing the verification result and messages
    """
    try:
        valid, messages = await verifier.verify(
            code=request.code, allow_sorry=request.allow_sorry
        )
        error_messages = [msg for msg in messages if msg["severity"] == "error"]
        error_str = get_error_str(erase_header(request.code), error_messages)
        return VerificationResponse(valid=valid, messages=messages, error_str=error_str)
    except Exception as e:
        logger.error(f"Error verifying code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_subgoals", response_model=ExtractSubgoalsResponse)
async def extract_subgoals(request: ExtractSubgoalsRequest):
    """
    Extract subgoals from a piece of Lean code.

    Args:
        request: The extract subgoals request containing the code

    Returns:
        ExtractSubgoalsResponse containing the subgoals
    """
    try:
        subgoals = await verifier.extract_subgoals(request.code)
        return ExtractSubgoalsResponse(subgoals=subgoals)
    except Exception as e:
        logger.error(f"Error extracting subgoals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the serve_verifier script."""
    import uvicorn

    lean_config = config.lean_server
    port = lean_config.get("LeanServerPort", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
