import argparse
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from theoremforge.utils import get_error_str
from theoremforge.lean_server.server import AsyncVerifier, erase_header
from theoremforge.config import Config


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


# Global variables to be initialized later
verifier = None
HEADER = None


def init_verifier(config_path: str):
    """Initialize the verifier with the given config path."""
    global verifier, HEADER

    config = Config(config_path)
    lean_config = config.lean_server

    HEADER = lean_config.get("LeanServerHeader", "import Mathlib\n")
    verifier = AsyncVerifier(lean_config)

    return config


def create_app(config_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if verifier is not None:
            verifier.initialize_worker_pool()
        yield
        # Shutdown
        if verifier is not None:
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
            valid, messages, error_str = await verifier.verify(
                code=request.code, allow_sorry=request.allow_sorry
            )

            return VerificationResponse(
                valid=valid, messages=messages, error_str=error_str
            )
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

    return app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lean Verifier API Server")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration file (config.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (overrides config file)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the serve_verifier script."""
    import uvicorn

    args = parse_args()

    # Initialize verifier with config
    config = init_verifier(args.config_path)

    # Get port from args or config
    lean_config = config.lean_server
    port = (
        args.port if args.port is not None else lean_config.get("LeanServerPort", 8000)
    )

    # Create app
    app = create_app(args.config_path)

    logger.info(f"Starting Lean Verifier API on {args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
