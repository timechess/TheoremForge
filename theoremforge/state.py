from pydantic import BaseModel
from typing import Optional, List
import asyncio

from theoremforge.lean_server.client import RemoteVerifier


class TheoremForgeContext:
    def __init__(self, verifier_url: str) -> None:
        self.verifier = RemoteVerifier(verifier_url)
        self.agents = dict()
        self.black_list = set()
        self.statement_record = dict()
        self.proof_record = dict()
        self.root_state_ids = set()  # Track root-level states (manually submitted)
        self.db = None  # Shared MongoDB client, initialized by manager

        # Locks for thread-safe concurrent access
        self.black_list_lock = asyncio.Lock()
        self.record_lock = asyncio.Lock()
        self.root_state_ids_lock = asyncio.Lock()


class TheoremForgeState(BaseModel):
    id: str
    informal_statement: Optional[str] = None
    normalized_statement: Optional[str] = None
    header: Optional[str] = "import Mathlib\n"
    formal_statement: Optional[str] = None
    formal_proof: Optional[str] = None
    informal_proof: Optional[str] = None
    proof_sketch: Optional[str] = None
    subgoals: Optional[List[str]] = None
    depth: int = 0
    parent_id: Optional[str] = None
    siblings: Optional[List[str]] = None
    success: bool = False
    metadata: Optional[dict] = {}
