"""
Async Queue Manager for TheoremForge.

This module provides a fully async queue manager that can continuously receive
new requests and process them concurrently.
"""

import asyncio
from typing import Optional, Callable, Any, Dict
from collections import defaultdict
from loguru import logger


class AsyncQueueManager:
    """
    A fully async queue manager that can continuously receive and process requests.
    
    Features:
    - Continuous request acceptance
    - Concurrent processing with configurable workers
    - Stage-based routing
    - Graceful shutdown
    - Real-time state persistence
    """
    
    def __init__(
        self, 
        max_workers: int = 10,
        state_callback: Optional[Callable[[Any], None]] = None
    ):
        """
        Initialize the async queue manager.
        
        Args:
            max_workers: Maximum number of concurrent workers per stage
            state_callback: Optional callback for state persistence
        """
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.max_workers = max_workers
        self.state_callback = state_callback
        self.workers: Dict[str, list] = defaultdict(list)
        self.running = False
        self.handlers: Dict[str, Callable] = {}
        self._shutdown_event = asyncio.Event()
        self._active_tasks = 0
        self._task_lock = asyncio.Lock()
        
    def register_handler(self, stage: str, handler: Callable):
        """Register a handler function for a specific stage."""
        self.handlers[stage] = handler
        logger.info(f"Registered handler for stage: {stage}")
        
    async def add_request(self, stage: str, state: Any) -> None:
        """
        Add a new request to the queue for a specific stage.
        
        Args:
            stage: The processing stage
            state: The state object to process
        """
        if not self.running:
            raise RuntimeError("Queue manager is not running. Call start() first.")
        
        await self.queues[stage].put(state)
        logger.debug(f"Added request to {stage} queue. Queue size: {self.queues[stage].qsize()}")
        
    async def start(self):
        """Start the queue manager and all workers."""
        if self.running:
            logger.warning("Queue manager is already running")
            return
            
        self.running = True
        self._shutdown_event.clear()
        
        # Start workers for each registered handler
        for stage in self.handlers:
            for i in range(self.max_workers):
                worker = asyncio.create_task(
                    self._worker(stage, i),
                    name=f"worker-{stage}-{i}"
                )
                self.workers[stage].append(worker)
                
        logger.info(f"Started queue manager with {self.max_workers} workers per stage")
        logger.info(f"Registered stages: {list(self.handlers.keys())}")
        
    async def stop(self, timeout: Optional[float] = None):
        """
        Gracefully stop the queue manager.
        
        Args:
            timeout: Maximum time to wait for workers to finish (seconds)
        """
        if not self.running:
            return
            
        logger.info("Stopping queue manager...")
        self.running = False
        self._shutdown_event.set()
        
        # Wait for all active tasks to complete
        try:
            if timeout:
                await asyncio.wait_for(
                    self._wait_for_completion(),
                    timeout=timeout
                )
            else:
                await self._wait_for_completion()
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for workers to finish")
            
        # Cancel all workers
        all_workers = []
        for worker_list in self.workers.values():
            all_workers.extend(worker_list)
            
        for worker in all_workers:
            if not worker.done():
                worker.cancel()
                
        # Wait for cancellation
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
            
        self.workers.clear()
        logger.info("Queue manager stopped")
        
    async def _wait_for_completion(self):
        """Wait for all queues to be empty and all tasks to complete."""
        while True:
            async with self._task_lock:
                if self._active_tasks == 0:
                    # Check if all queues are empty
                    all_empty = all(
                        queue.qsize() == 0 
                        for queue in self.queues.values()
                    )
                    if all_empty:
                        break
            await asyncio.sleep(0.1)
            
    async def _worker(self, stage: str, worker_id: int):
        """
        Worker coroutine that processes items from a specific stage queue.
        
        Args:
            stage: The stage this worker handles
            worker_id: Unique identifier for this worker
        """
        logger.debug(f"Worker {worker_id} started for stage: {stage}")
        
        while self.running or not self.queues[stage].empty():
            try:
                # Try to get an item with timeout
                try:
                    state = await asyncio.wait_for(
                        self.queues[stage].get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check if we should shutdown
                    if self._shutdown_event.is_set():
                        break
                    continue
                    
                # Increment active task counter
                async with self._task_lock:
                    self._active_tasks += 1
                    
                try:
                    # Process the state
                    logger.debug(f"Worker {worker_id} processing state {state.id} at stage {stage}")
                    new_state, metadata = await self.handlers[stage](state)
                    
                    # Route to next stage
                    if new_state.stage != stage and new_state.stage != "finished":
                        await self.add_request(new_state.stage, new_state)
                    elif new_state.stage == "finished":
                        # Persist finished state
                        if self.state_callback:
                            try:
                                await self.state_callback(new_state)
                            except Exception as e:
                                logger.error(f"Error in state callback: {e}")
                                
                    logger.debug(f"Worker {worker_id} completed state {state.id}")
                    
                except Exception as e:
                    logger.error(f"Error processing state {state.id} at stage {stage}: {e}")
                    # Mark as failed and persist
                    state.stage = "finished"
                    state.result = "failure"
                    if self.state_callback:
                        try:
                            await self.state_callback(state)
                        except Exception as e2:
                            logger.error(f"Error in state callback: {e2}")
                finally:
                    # Decrement active task counter
                    async with self._task_lock:
                        self._active_tasks -= 1
                    self.queues[stage].task_done()
                    
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} for stage {stage} cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_id} for stage {stage}: {e}")
                
        logger.debug(f"Worker {worker_id} stopped for stage: {stage}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            "running": self.running,
            "active_tasks": self._active_tasks,
            "queue_sizes": {
                stage: queue.qsize() 
                for stage, queue in self.queues.items()
            },
            "workers_per_stage": {
                stage: len(workers) 
                for stage, workers in self.workers.items()
            }
        }

