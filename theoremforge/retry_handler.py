"""
Retry handler for TheoremForge operations.

Provides configurable retry logic with exponential backoff for handling
transient failures in agent operations.
"""

import asyncio
from typing import Callable, Any, Optional, Type
from loguru import logger
import traceback


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retry_on_exceptions: tuple = (Exception,)
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            retry_on_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_exceptions = retry_on_exceptions


class RetryHandler:
    """
    Handler for retrying failed operations with exponential backoff.
    
    Features:
    - Configurable retry attempts
    - Exponential backoff
    - Selective exception handling
    - Detailed logging
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                # Success - reset any backoff and return
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except self.config.retry_on_exceptions as e:
                last_exception = e
                
                # Check if we should retry
                if attempt < self.config.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.config.initial_delay * (self.config.exponential_base ** attempt),
                        self.config.max_delay
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {str(e)}"
                    )
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(
                        f"All {self.config.max_retries + 1} attempts failed. "
                        f"Last error: {str(e)}"
                    )
                    logger.debug(f"Final exception details: {traceback.format_exc()}")
                    raise
                    
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception occurred: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                raise
                
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily disabling operations
    that are consistently failing.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery (seconds)
            half_open_attempts: Number of test attempts in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.
        
        Args:
            func: The async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The result of the function
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self.lock:
            # Check circuit state
            if self.state == "open":
                # Check if we should try recovery
                if (self.last_failure_time and 
                    asyncio.get_event_loop().time() - self.last_failure_time >= self.recovery_timeout):
                    self.state = "half-open"
                    self.failure_count = 0
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise Exception("Circuit breaker is OPEN - operation not attempted")
                    
        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Success - update state
            async with self.lock:
                if self.state == "half-open":
                    self.state = "closed"
                    logger.info("Circuit breaker closed after successful recovery")
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            # Failure - update state
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = asyncio.get_event_loop().time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker OPENED after {self.failure_count} failures"
                    )
                elif self.state == "half-open":
                    self.state = "open"
                    logger.warning("Circuit breaker re-opened during recovery attempt")
                    
            raise


# Convenience function for simple retry
async def retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    **kwargs
) -> Any:
    """
    Simple retry function for one-off operations.
    
    Args:
        func: Function to retry
        *args: Positional arguments
        max_retries: Maximum retry attempts
        initial_delay: Initial delay between retries
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    handler = RetryHandler(RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay
    ))
    return await handler.execute_with_retry(func, *args, **kwargs)

