from pydantic import BaseModel
from typing import Optional, List
import asyncio
import multiprocessing as mp
from multiprocessing.managers import SyncManager

from theoremforge.db import SQLiteClient
from theoremforge.lean_server.client import RemoteVerifier


class TheoremForgeContext:
    """
    Shared context for TheoremForge agents.
    
    Supports two modes:
    1. Single-process mode (default): Uses regular Python data structures
    2. Multi-process mode: Uses multiprocessing.Manager for shared state
       when finish_agent runs in a separate process
    """
    
    def __init__(self, verifier_config, mp_manager: Optional[SyncManager] = None) -> None:
        self.verifier = RemoteVerifier(
            f"http://localhost:{verifier_config.get('LeanServerPort')}"
        )
        self.agents = dict()
        
        # Multi-process mode: use Manager proxies for shared state
        self._mp_manager = mp_manager
        self._mp_mode = mp_manager is not None
        
        if self._mp_mode:
            # Multiprocess-safe shared data structures
            # Use dict to simulate set for O(1) lookups (key -> True)
            self.black_list = mp_manager.dict()  # state_id -> True (simulates set)
            self.statement_record = mp_manager.dict()
            self.proof_record = mp_manager.dict()
            self.root_state_ids = mp_manager.dict()  # state_id -> True (simulates set)
            self.cancellation_flags = mp_manager.dict()  # state_id -> bool
            # For backward compatibility, create empty dict (not used in mp mode)
            self.cancellation_events = {}
        else:
            # Single-process mode: use regular Python data structures
            self.black_list = set()
            self.statement_record = dict()
            self.proof_record = dict()
            self.root_state_ids = set()
            self.cancellation_events = dict()  # state_id -> asyncio.Event
            self.cancellation_flags = None  # Not used in single-process mode
        
        self.db: SQLiteClient | None = None
        
        # Shared task queues per agent type
        # In mp mode, finish_agent uses mp.Queue, others use asyncio.Queue
        self.shared_queues = dict()  # agent_name -> asyncio.Queue or mp.Queue
        
        # mp.Queue for finish_agent (created separately in mp mode)
        self.finish_queue: Optional[mp.Queue] = None

        # Locks for thread-safe concurrent access (only used in single-process mode)
        self.black_list_lock = asyncio.Lock()
        self.record_lock = asyncio.Lock()
        self.root_state_ids_lock = asyncio.Lock()
        self.cancellation_lock = asyncio.Lock()
    
    def is_mp_mode(self) -> bool:
        """Check if running in multi-process mode."""
        return self._mp_mode
    
    # Helper methods for multiprocess-safe operations
    def add_to_blacklist(self, state_id: str):
        """Add state to blacklist (works in both modes). O(1) in both modes."""
        if self._mp_mode:
            self.black_list[state_id] = True  # dict as set
        else:
            self.black_list.add(state_id)
    
    def is_blacklisted(self, state_id: str, parent_id: Optional[str] = None) -> bool:
        """Check if state is blacklisted (works in both modes). O(1) in both modes."""
        if self._mp_mode:
            return state_id in self.black_list or (parent_id and parent_id in self.black_list)
        else:
            return state_id in self.black_list or (parent_id and parent_id in self.black_list)
    
    def set_cancelled(self, state_id: str):
        """Mark state as cancelled (works in both modes)."""
        if self._mp_mode:
            self.cancellation_flags[state_id] = True
        else:
            if state_id not in self.cancellation_events:
                self.cancellation_events[state_id] = asyncio.Event()
            self.cancellation_events[state_id].set()
    
    def is_cancelled(self, state_id: str, parent_id: Optional[str] = None) -> bool:
        """Check if state is cancelled (works in both modes)."""
        if self._mp_mode:
            if self.cancellation_flags.get(state_id, False):
                return True
            if parent_id and self.cancellation_flags.get(parent_id, False):
                return True
            return False
        else:
            event = self.cancellation_events.get(state_id)
            if event is not None and event.is_set():
                return True
            if parent_id:
                parent_event = self.cancellation_events.get(parent_id)
                if parent_event is not None and parent_event.is_set():
                    return True
            return False
    
    def cleanup_cancellation(self, state_id: str):
        """Remove cancellation for a state (works in both modes)."""
        if self._mp_mode:
            self.cancellation_flags.pop(state_id, None)
        else:
            self.cancellation_events.pop(state_id, None)
    
    def add_root_state(self, state_id: str):
        """Add root state (works in both modes). O(1) in both modes."""
        if self._mp_mode:
            self.root_state_ids[state_id] = True  # dict as set
        else:
            self.root_state_ids.add(state_id)
    


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
    token_trace: Optional[dict] = {}
    metadata: Optional[dict] = {}
