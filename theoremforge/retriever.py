from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
from loguru import logger

from lean_explore.search.service import Service, SearchResult

class Retriever:
    """
    Async-compatible retriever with parallel search capabilities.
    
    Uses a single shared Service instance with thread-safe access.
    ThreadPoolExecutor is used for parallel query dispatch while
    ensuring thread-safe access to the shared Service.
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        """
        Initialize the retriever with a thread pool and shared Service.
        
        Args:
            max_workers: Maximum number of parallel workers for search operations
        """
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._async_lock = asyncio.Lock()
        self._closed = False
        
        # Single shared Service instance with thread lock for safe access
        self._service = Service()
        self._service_lock = threading.Lock()
        
        logger.info(f"Retriever initialized with max_workers={max_workers}, using single shared Service")
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="retriever_worker"
            )
        return self._executor

    def _search_single_query(self, query: str, sources: List[str], topk: int) -> List[SearchResult]:
        """
        Thread-safe search for a single query using the shared Service.
        
        Args:
            query: Search query string
            sources: List of sources to search
            topk: Number of top results to return
            
        Returns:
            List of search results for the query
        """
        try:
            # Thread-safe access to shared Service
            with self._service_lock:
                return self._service.search(query, sources, topk).results
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return []

    def search(self, queries: List[str], topk: int) -> List[SearchResult]:
        """
        Synchronous search across multiple queries (legacy interface).
        
        Args:
            queries: List of search queries
            topk: Number of top results to return per query
            
        Returns:
            Flattened list of all search results
        """
        if not queries:
            return []
        
        if self._closed:
            raise RuntimeError("Retriever has been closed")
        
        sources = ["Mathlib", "Init"]
        executor = self._get_executor()
        
        # Execute searches in parallel (each acquiring the lock as needed)
        try:
            futures = [
                executor.submit(self._search_single_query, query, sources, topk)
                for query in queries
            ]
            results = [f.result() for f in futures]
            # Flatten results
            return sum(results, [])
        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            # Fallback to sequential search
            return self._search_sequential(queries, sources, topk)
    
    def _search_sequential(
        self, queries: List[str], sources: List[str], topk: int
    ) -> List[SearchResult]:
        """Fallback sequential search when parallel execution fails."""
        search_results = []
        for query in queries:
            results = self._search_single_query(query, sources, topk)
            search_results.extend(results)
        return search_results

    async def search_async(
        self, queries: List[str], topk: int
    ) -> List[SearchResult]:
        """
        Asynchronous search across multiple queries.
        
        This method runs searches using a thread pool executor,
        allowing the async event loop to continue processing other tasks.
        All searches use the shared Service instance with thread-safe access.
        
        Args:
            queries: List of search queries
            topk: Number of top results to return per query
            
        Returns:
            Flattened list of all search results
        """
        if not queries:
            return []
        
        if self._closed:
            raise RuntimeError("Retriever has been closed")
        
        sources = ["Mathlib", "Init"]
        loop = asyncio.get_event_loop()
        executor = self._get_executor()
        
        # Create tasks for parallel execution
        async def search_single(query: str) -> List[SearchResult]:
            return await loop.run_in_executor(
                executor,
                self._search_single_query,
                query, sources, topk
            )
        
        # Execute all searches concurrently
        try:
            tasks = [search_single(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and flatten results
            flattened = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Async search failed for query '{queries[i]}': {result}")
                else:
                    flattened.extend(result)
            return flattened
            
        except Exception as e:
            logger.error(f"Async parallel search failed: {e}")
            # Fallback to synchronous search in executor
            return await loop.run_in_executor(
                executor,
                lambda: self._search_sequential(queries, sources, topk)
            )

    async def close(self) -> None:
        """
        Close the retriever and release resources.
        
        This method should be called when the retriever is no longer needed
        to properly shut down the thread pool.
        """
        async with self._async_lock:
            if self._closed:
                return
            
            self._closed = True
            
            if self._executor is not None:
                logger.info("Shutting down retriever thread pool...")
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None
                logger.info("Retriever thread pool shut down complete")

    def __del__(self):
        """Ensure thread pool is cleaned up on garbage collection."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
