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
        # Note: finish_agent queue is handled separately in mp mode
        if agent_name not in self.context.shared_queues and agent_name != "finish_agent":
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
                logger.error(f"{self.agent_name}: {e}")
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
        
        For finish_agent in mp mode, uses multiprocessing.Queue.
        """
        logger.debug(f"Adding state {state.id} to {agent_name} shared queue")
        
        if agent_name == "finish_agent" and self.context.is_mp_mode():
            # In multiprocess mode, send to finish_agent via mp.Queue
            # Serialize state to dict for cross-process transfer
            self.context.finish_queue.put(state.model_dump())
        else:
            await self.context.shared_queues[agent_name].put(state)

    async def is_cancelled(self, state: TheoremForgeState) -> bool:
        """
        Check if a state has been cancelled.
        
        Works in both single-process and multi-process modes.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state or its parent is cancelled, False otherwise
        """
        return self.context.is_cancelled(state.id, state.parent_id)

    async def is_blacklisted(self, state: TheoremForgeState) -> bool:
        """
        Check if a state is blacklisted.
        
        Works in both single-process and multi-process modes.
        
        Args:
            state: The state to check
            
        Returns:
            True if the state or its parent is blacklisted, False otherwise
        """
        return self.context.is_blacklisted(state.id, state.parent_id)

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
            logger.debug(
                f"{self.agent_name}: State {state.id} is cancelled, skipping"
            )
            return True
        
        return False

    async def register_cancellation_event(self, state: TheoremForgeState):
        """
        Register a cancellation event for a state before starting work.
        
        Works in both single-process and multi-process modes.
        
        Args:
            state: The state to register
        """
        if self.context.is_mp_mode():
            # In mp mode, just ensure the flag exists (False = not cancelled)
            if state.id not in self.context.cancellation_flags:
                self.context.cancellation_flags[state.id] = False
        else:
            if state.id not in self.context.cancellation_events:
                self.context.cancellation_events[state.id] = asyncio.Event()

    async def cleanup_cancellation_event(self, state: TheoremForgeState):
        """
        Clean up cancellation event for a state after work is done.
        
        Works in both single-process and multi-process modes.
        
        Args:
            state: The state to clean up
        """
        self.context.cleanup_cancellation(state.id)
