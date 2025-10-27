import asyncio
from typing import List, Callable, Any, Union
from enum import Enum


class JobExecutionMode(Enum):
    """Enum defining the different job execution modes."""

    WAIT_FOR_ALL = "wait_for_all"
    FIRST_SUCCESS = "first_success"
    FIRST_FAILURE = "first_failure"


class AsyncJobPool:
    """
    An asyncio-based job pool that can execute jobs with different termination strategies.

    Supports three execution modes:
    1. WAIT_FOR_ALL: Waits for all jobs to complete before terminating
    2. FIRST_SUCCESS: Terminates other jobs when the first one succeeds
    3. FIRST_FAILURE: Terminates all jobs when the first one fails
    """

    def __init__(self):
        self._running_tasks: List[asyncio.Task] = []
        self._results: List[Any] = []
        self._exceptions: List[Exception] = []

    async def execute_jobs(
        self,
        job_func: Callable,
        job_params: List[Union[tuple, dict]],
        mode: Union[JobExecutionMode, str],
        result_collector: Callable[[List[Any], List[Exception]], Any] = None,
    ) -> dict:
        """
        Execute the same function multiple times with different parameters according to the specified mode.

        Args:
            job_func: The function to execute multiple times
            job_params: List of parameters for each job execution. Each item can be:
                       - tuple: positional arguments (*args)
                       - dict: keyword arguments (**kwargs)
                       - dict with 'args' and 'kwargs' keys for both
            mode: Execution mode (JobExecutionMode enum or string)
            result_collector: Optional function to customize result collection.
                            Takes (results: List[Any], exceptions: List[Exception]) -> Any
                            If None, returns default dict format.

        Returns:
            dict: Results containing 'results', 'exceptions', 'completed_count', 'cancelled_count'
                  If result_collector is provided, 'custom_result' key is added with its output.
        """
        if isinstance(mode, str):
            try:
                mode = JobExecutionMode(mode)
            except ValueError:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be one of {[m.value for m in JobExecutionMode]}"
                )

        if not job_params:
            return {
                "results": [],
                "exceptions": [],
                "completed_count": 0,
                "cancelled_count": 0,
                "mode": mode.value,
            }

        # Reset state
        self._running_tasks = []
        self._results = []
        self._exceptions = []

        # Create tasks for all job parameter sets
        for i, params in enumerate(job_params):
            task = asyncio.create_task(
                self._execute_single_job(job_func, params), name=f"job_{i}"
            )
            self._running_tasks.append(task)

        try:
            if mode == JobExecutionMode.WAIT_FOR_ALL:
                result = await self._wait_for_all()
            elif mode == JobExecutionMode.FIRST_SUCCESS:
                result = await self._first_success()
            elif mode == JobExecutionMode.FIRST_FAILURE:
                result = await self._first_failure()

            # Apply custom result collector if provided
            if result_collector:
                result["custom_result"] = result_collector(
                    result["results"], result["exceptions"]
                )

            return result
        finally:
            # Ensure all tasks are cancelled if not already done
            await self._cleanup_tasks()

    async def _execute_single_job(
        self, job_func: Callable, params: Union[tuple, dict]
    ) -> Any:
        """Execute a single job with the given parameters, handling both sync and async cases."""
        # Parse parameters
        if isinstance(params, dict):
            if "args" in params and "kwargs" in params:
                # Dict with both args and kwargs
                args = (
                    params["args"]
                    if isinstance(params["args"], (tuple, list))
                    else (params["args"],)
                )
                kwargs = params["kwargs"] if isinstance(params["kwargs"], dict) else {}
            else:
                # Dict is kwargs
                args = ()
                kwargs = params
        elif isinstance(params, (tuple, list)):
            # Tuple/list is args
            args = params
            kwargs = {}
        else:
            # Single parameter
            args = (params,)
            kwargs = {}

        # Call the job function
        job_result = job_func(*args, **kwargs)

        if asyncio.iscoroutine(job_result):
            # If calling the job returned a coroutine, await it
            return await job_result
        else:
            # If it's not a coroutine, return the result directly
            return job_result

    async def _run_sync_job(self, job: Callable, *args, **kwargs) -> Any:
        """Wrapper to run synchronous jobs in the event loop."""
        return await asyncio.get_event_loop().run_in_executor(
            None, job, *args, **kwargs
        )

    async def _wait_for_all(self) -> dict:
        """Wait for all jobs to complete, collecting all results and exceptions."""
        completed_count = 0
        cancelled_count = 0

        # Wait for all tasks to complete
        results = await asyncio.gather(*self._running_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._exceptions.append(result)
            else:
                self._results.append(result)
            completed_count += 1

        return {
            "results": self._results,
            "exceptions": self._exceptions,
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "mode": JobExecutionMode.WAIT_FOR_ALL.value,
        }

    async def _first_success(self) -> dict:
        """Terminate other jobs when the first one succeeds."""
        completed_count = 0
        cancelled_count = 0

        try:
            # Wait for the first task to complete successfully
            done, pending = await asyncio.wait(
                self._running_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check completed tasks for success
            success_found = False
            for task in done:
                completed_count += 1
                try:
                    result = await task
                    self._results.append(result)
                    success_found = True
                    break  # First success found
                except Exception as e:
                    self._exceptions.append(e)

            if success_found:
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    cancelled_count += 1

                # Wait for cancellation to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            else:
                # No success yet, continue with remaining tasks
                return await self._continue_first_success(
                    pending, completed_count, cancelled_count
                )

        except Exception as e:
            self._exceptions.append(e)

        return {
            "results": self._results,
            "exceptions": self._exceptions,
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "mode": JobExecutionMode.FIRST_SUCCESS.value,
        }

    async def _continue_first_success(
        self, remaining_tasks: set, completed_count: int, cancelled_count: int
    ) -> dict:
        """Continue waiting for first success among remaining tasks."""
        while remaining_tasks:
            done, pending = await asyncio.wait(
                remaining_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                completed_count += 1
                try:
                    result = await task
                    self._results.append(result)
                    # Success found, cancel remaining
                    for remaining_task in pending:
                        remaining_task.cancel()
                        cancelled_count += 1

                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    return {
                        "results": self._results,
                        "exceptions": self._exceptions,
                        "completed_count": completed_count,
                        "cancelled_count": cancelled_count,
                        "mode": JobExecutionMode.FIRST_SUCCESS.value,
                    }
                except Exception as e:
                    self._exceptions.append(e)

            remaining_tasks = pending

        # All tasks completed without success
        return {
            "results": self._results,
            "exceptions": self._exceptions,
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "mode": JobExecutionMode.FIRST_SUCCESS.value,
        }

    async def _first_failure(self) -> dict:
        """Terminate all jobs when the first one fails."""
        completed_count = 0
        cancelled_count = 0

        try:
            # Wait for the first task to complete
            done, pending = await asyncio.wait(
                self._running_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check completed tasks for failure
            failure_found = False
            for task in done:
                completed_count += 1
                try:
                    result = await task
                    self._results.append(result)
                except Exception as e:
                    self._exceptions.append(e)
                    failure_found = True
                    break  # First failure found

            if failure_found:
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    cancelled_count += 1

                # Wait for cancellation to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            else:
                # No failure yet, continue with remaining tasks
                return await self._continue_first_failure(
                    pending, completed_count, cancelled_count
                )

        except Exception as e:
            self._exceptions.append(e)

        return {
            "results": self._results,
            "exceptions": self._exceptions,
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "mode": JobExecutionMode.FIRST_FAILURE.value,
        }

    async def _continue_first_failure(
        self, remaining_tasks: set, completed_count: int, cancelled_count: int
    ) -> dict:
        """Continue waiting for first failure among remaining tasks."""
        while remaining_tasks:
            done, pending = await asyncio.wait(
                remaining_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                completed_count += 1
                try:
                    result = await task
                    self._results.append(result)
                except Exception as e:
                    self._exceptions.append(e)
                    # Failure found, cancel remaining
                    for remaining_task in pending:
                        remaining_task.cancel()
                        cancelled_count += 1

                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    return {
                        "results": self._results,
                        "exceptions": self._exceptions,
                        "completed_count": completed_count,
                        "cancelled_count": cancelled_count,
                        "mode": JobExecutionMode.FIRST_FAILURE.value,
                    }

            remaining_tasks = pending

        # All tasks completed without failure
        return {
            "results": self._results,
            "exceptions": self._exceptions,
            "completed_count": completed_count,
            "cancelled_count": cancelled_count,
            "mode": JobExecutionMode.FIRST_FAILURE.value,
        }

    async def _cleanup_tasks(self):
        """Cancel any remaining running tasks."""
        for task in self._running_tasks:
            if not task.done():
                task.cancel()

        # Wait for all cancellations to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
