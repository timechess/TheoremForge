from theoremforge.state import TheoremForgeState, TheoremForgeContext
from abc import abstractmethod
import asyncio
from loguru import logger
from theoremforge.db import SQLiteClient
from theoremforge.utils import CancellationError


class BaseAgent:
    def __init__(self, agent_name: str, context: TheoremForgeContext, *args, **kwargs):
        self.agent_name = agent_name
        self.context = context
        
        # Create shared queue for this agent type if it doesn't exist
        if agent_name not in self.context.shared_queues:
            self.context.shared_queues[agent_name] = asyncio.Queue()

    @property
    def db(self) -> SQLiteClient:
        """Access shared database client from context."""
        return self.context.db
    
    @property
    def task_queue(self) -> asyncio.Queue[TheoremForgeState]:
        """Access shared task queue for this agent type."""
        return self.context.shared_queues[self.agent_name]

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    continue

                await self._run(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"{self.agent_name}: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
            except Exception as e:
                logger.error(f"{self.agent_name}: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
                await asyncio.sleep(1)

    @abstractmethod
    async def _run(self, state: TheoremForgeState):
        raise NotImplementedError

    async def add_state_request(self, agent_name: str, state: TheoremForgeState):
        """
        Add a state to the shared queue of the specified agent type.
        All agents of that type will compete to process it.
        """
        logger.debug(f"Adding state {state.id} to {agent_name} shared queue")
        await self.context.shared_queues[agent_name].put(state)

    async def is_cancelled(self, state: TheoremForgeState) -> bool:
        """
        Check if a state has been cancelled.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state or its parent is cancelled, False otherwise
        """
        async with self.context.cancellation_lock:
            # Check if this state is cancelled
            if state.id in self.context.cancellation_events:
                if self.context.cancellation_events[state.id].is_set():
                    return True
            
            # Check if parent is cancelled
            if state.parent_id and state.parent_id in self.context.cancellation_events:
                if self.context.cancellation_events[state.parent_id].is_set():
                    return True
        
        return False

    async def is_blacklisted(self, state: TheoremForgeState) -> bool:
        """
        Check if a state is blacklisted.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state or its parent is blacklisted, False otherwise
        """
        async with self.context.black_list_lock:
            return (
                state.id in self.context.black_list 
                or (state.parent_id and state.parent_id in self.context.black_list)
            )

    async def should_skip_state(self, state: TheoremForgeState) -> bool:
        """
        Check if a state should be skipped (blacklisted or cancelled).
        
        Args:
            state: The state to check
            
        Returns:
            True if the state should be skipped, False otherwise
        """
        if await self.is_blacklisted(state):
            logger.debug(
                f"{self.agent_name}: State {state.id} is blacklisted, skipping"
            )
            return True
        
        if await self.is_cancelled(state):
            logger.info(
                f"{self.agent_name}: State {state.id} is cancelled, skipping"
            )
            return True
        
        return False

    async def register_cancellation_event(self, state: TheoremForgeState):
        """
        Register a cancellation event for a state before starting work.
        
        Args:
            state: The state to register
        """
        async with self.context.cancellation_lock:
            if state.id not in self.context.cancellation_events:
                self.context.cancellation_events[state.id] = asyncio.Event()

    async def cleanup_cancellation_event(self, state: TheoremForgeState):
        """
        Clean up cancellation event for a state after work is done.
        
        Args:
            state: The state to clean up
        """
        async with self.context.cancellation_lock:
            if state.id in self.context.cancellation_events:
                del self.context.cancellation_events[state.id]
