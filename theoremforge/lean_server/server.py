from lean_interact import (
    LeanREPLConfig,
    AutoLeanServer,
    LocalProject,
    Command,
)
from lean_interact.interface import LeanError
from loguru import logger
import re
import asyncio
import multiprocessing
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from theoremforge.config import config

lean_config = config.lean_server
HEADER = lean_config.get("LeanServerHeader", "import Mathlib\n")
DEFAULT_TIMEOUT = lean_config.get("LeanServerTimeout", 20)


def erase_header(code: str) -> str:
    """
    Remove any import statements from the Lean code.

    Args:
        code (str): The Lean source code to process.

    Returns:
        str: The code with all import statements removed.
    """
    import_pattern = re.compile(r"^import\s+.*$", re.MULTILINE)
    open_pattern = re.compile(r"^open\s+.*$", re.MULTILINE)
    set_option_pattern = re.compile(r"^set_option\s+.*$", re.MULTILINE)
    return re.sub(
        import_pattern,
        "",
        re.sub(open_pattern, "", re.sub(set_option_pattern, "", code)),
    ).strip("\n")


def normalize_header(code: str) -> str:
    """
    Replace all import statements in the code with the standard header.

    Args:
        code (str): The Lean source code to normalize.

    Returns:
        str: The code with standardized import statements.
    """
    import_pattern = re.compile(r"^import\s+.*$", re.MULTILINE)
    open_pattern = re.compile(r"^open\s+.*$", re.MULTILINE)
    set_option_pattern = re.compile(r"^set_option\s+.*$", re.MULTILINE)
    normalized_code = re.sub(
        import_pattern,
        HEADER,
        re.sub(open_pattern, "", re.sub(set_option_pattern, "", code)),
    )
    return normalized_code


class LeanServerWorker(multiprocessing.Process):
    """
    A worker process that maintains a persistent Lean server instance.

    This worker runs in a separate process and handles verification tasks
    from a queue, maintaining its own Lean server to maximize performance
    by avoiding server initialization overhead for each task.

    Attributes:
        worker_id (int): Unique identifier for this worker.
        task_queue (multiprocessing.Queue): Queue from which tasks are received.
        result_queue (multiprocessing.Queue): Queue where results are placed.
        repl_config (LeanREPLConfig): Configuration for the Lean server.
    """

    def __init__(self, worker_id, task_queue, result_queue, repl_config, init_event):
        """
        Initialize a new worker process.

        Args:
            worker_id (int): Unique identifier for this worker.
            task_queue (multiprocessing.Queue): Queue from which tasks are received.
            result_queue (multiprocessing.Queue): Queue where results are placed.
            repl_config (LeanREPLConfig): Configuration for the Lean server.
            init_event (multiprocessing.Event): Event to signal when worker is initialized.
        """
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.repl_config = repl_config
        self.init_event = init_event

    def run(self):
        """
        Main worker process function that initializes a Lean server and processes tasks.

        This method runs in a separate process and:
        1. Initializes a Lean server with the standard header
        2. Continuously pulls tasks from the task queue
        3. Processes each task using the Lean server
        4. Puts results in the result queue
        5. Handles errors and attempts to reinitialize the server on failure
        """
        logger.info(f"Worker {self.worker_id}: Initializing Lean server")
        try:
            server = AutoLeanServer(self.repl_config)
            context_env = server.run(Command(cmd=HEADER), add_to_session_cache=True).env
            logger.info(f"Worker {self.worker_id}: Lean server initialized")

            # Signal that this worker is fully initialized
            self.init_event.set()

            while True:
                task_id, args, task_type = self.task_queue.get()
                if task_id is None:
                    logger.info(f"Worker {self.worker_id}: Shutting down")
                    break

                if task_type == "verify":
                    try:
                        logger.debug(
                            f"Worker {self.worker_id}: Processing task {task_id}"
                        )
                        code, allow_sorry = args
                        response = server.run(
                            Command(cmd=erase_header(code), env=context_env),
                            timeout=DEFAULT_TIMEOUT,
                        )

                        if isinstance(response, LeanError):
                            logger.error(
                                f"Worker {self.worker_id}: Lean error: {response}"
                            )
                            self.result_queue.put((task_id, (False, [])))
                        else:
                            is_valid = response.lean_code_is_valid(
                                allow_sorry=allow_sorry
                            )
                            messages = [msg.model_dump() for msg in response.messages]
                            self.result_queue.put((task_id, (is_valid, messages)))

                    except Exception as e:
                        logger.warning(f"Worker {self.worker_id}: Error: {e}")
                        # Try to reinitialize the server
                        try:
                            logger.info(
                                f"Worker {self.worker_id}: Reinitializing server"
                            )
                            server = AutoLeanServer(self.repl_config)
                            context_env = server.run(
                                Command(cmd=HEADER), add_to_session_cache=True
                            ).env
                        except Exception as e2:
                            logger.error(
                                f"Worker {self.worker_id}: Failed to reinitialize: {e2}"
                            )

                        self.result_queue.put((task_id, (False, [])))
                elif task_type == "extract_subgoals":
                    try:
                        logger.debug(
                            f"Worker {self.worker_id}: Processing task {task_id}"
                        )
                        code = args
                        response = server.run(
                            Command(cmd=erase_header(code), env=context_env),
                            timeout=DEFAULT_TIMEOUT,
                        )
                        processed_code = code.replace("sorry", "extract_goal")
                        sorries = response.sorries
                        processed_response = server.run(
                            Command(cmd=erase_header(processed_code), env=context_env),
                            timeout=DEFAULT_TIMEOUT,
                        )
                        extracted_goals = {
                            (
                                goal.start_pos.line,
                                goal.start_pos.column,
                            ): goal.data.strip()
                            for goal in processed_response.messages
                        }
                        subgoals = []
                        for sorry in sorries:
                            subgoals.append(
                                extracted_goals[
                                    (sorry.start_pos.line, sorry.start_pos.column)
                                ]
                            )
                        self.result_queue.put((task_id, subgoals))
                    except Exception as e:
                        logger.warning(f"Worker {self.worker_id}: Error: {e}")
                        self.result_queue.put((task_id, []))
                else:
                    logger.warning(
                        f"Worker {self.worker_id}: Unknown task type: {task_type}"
                    )
                    self.result_queue.put((task_id, []))
                logger.debug(f"Worker {self.worker_id}: Task {task_id} processed")

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Fatal error: {e}")


class AsyncVerifier:
    """
    An asynchronous verifier for Lean code with parallel processing capabilities.

    This class provides methods to verify Lean code correctness using multiple
    worker processes, each with its own Lean server instance. It is designed for
    high-throughput parallel verification tasks.

    Attributes:
        repl_config (LeanREPLConfig): Configuration for the Lean servers.
        workers (int): Number of worker processes to use.
        task_queue (multiprocessing.Queue): Queue for distributing tasks to workers.
        result_queue (multiprocessing.Queue): Queue for collecting results from workers.
        worker_processes (list): List of worker process instances.
        next_task_id (int): Counter for generating unique task IDs.
        is_initialized (bool): Whether the worker pool has been initialized.
    """

    def __init__(self, project: str, workers: int = 8) -> None:
        """
        Initialize a new AsyncVerifier.

        Args:
            project (str): The project to use for verification.
            workers (int, optional): Number of worker processes to use. Defaults to the number of CPU cores.
        """
        self.repl_config = LeanREPLConfig(project=LocalProject(directory=project))
        self.workers = workers
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker_processes = []
        self.next_task_id = 0
        self.is_initialized = False
        self.init_events = []

        logger.info(f"AsyncVerifier initialized with {workers} workers")

    def initialize_worker_pool(self):
        """
        Initialize the worker pool with persistent server processes.

        This method creates and starts worker processes, each with its own Lean server.
        The servers are initialized once and persist for the lifetime of the worker,
        which avoids the overhead of initializing a new server for each verification task.
        """
        if self.is_initialized:
            return

        logger.info(f"Starting {self.workers} worker processes with persistent servers")

        # Create initialization events for each worker
        self.init_events = [multiprocessing.Event() for _ in range(self.workers)]

        # Create and start worker processes
        for i in range(self.workers):
            worker = LeanServerWorker(
                worker_id=i + 1,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                repl_config=self.repl_config,
                init_event=self.init_events[i],
            )
            worker.start()
            self.worker_processes.append(worker)

        # Wait for all workers to be fully initialized
        logger.info("Waiting for all workers to initialize their Lean servers...")
        for i, init_event in enumerate(self.init_events):
            init_event.wait()
            logger.info(f"Worker {i + 1} initialization confirmed")

        self.is_initialized = True
        logger.info("Worker pool initialization complete - all workers ready")

    def shutdown(self):
        """
        Shutdown the worker pool.

        This method sends termination signals to all worker processes and
        waits for them to finish, with a timeout to avoid hanging.
        """
        if not self.is_initialized:
            return

        logger.info("Shutting down worker pool")
        for _ in range(len(self.worker_processes)):
            self.task_queue.put((None, None, None))

        for worker in self.worker_processes:
            worker.join(timeout=5)
            if worker.is_alive():
                logger.warning(
                    f"Worker {worker.worker_id} did not terminate gracefully"
                )

        self.worker_processes = []
        self.is_initialized = False
        logger.info("Worker pool shutdown complete")

    async def verify(
        self, code: str, allow_sorry: bool = True
    ) -> tuple[bool, list[dict]]:
        """
        Asynchronously verify a single piece of Lean code.

        This method submits the code to a worker process and asynchronously waits for the result.

        Args:
            code (str): The Lean code to verify.
            allow_sorry (bool, optional): Whether to allow 'sorry' in the code. Defaults to True.

        Returns:
            tuple[bool, list[dict]]: A tuple containing:
                - bool: Whether the verification was successful.
                - list[dict]: Messages from the verification process.
        """
        if not self.is_initialized:
            self.initialize_worker_pool()

        task_id = self.next_task_id
        self.next_task_id += 1

        self.task_queue.put((task_id, (code, allow_sorry), "verify"))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async def wait_for_result():
            while True:
                try:
                    result_task_id, result = self.result_queue.get_nowait()
                    if result_task_id == task_id:
                        future.set_result(result)
                        return
                    else:
                        self.result_queue.put((result_task_id, result))
                except Exception:
                    await asyncio.sleep(0.01)

        asyncio.create_task(wait_for_result())

        return await future

    async def batched_verify(
        self, codes: list[str], allow_sorry: bool = True
    ) -> list[tuple[bool, list[dict]]]:
        """
        Asynchronously verify multiple pieces of Lean code in parallel.

        This method submits all codes to worker processes and waits for all results.
        The verification is performed in parallel across multiple worker processes,
        each with its own persistent Lean server.

        Args:
            codes (list[str]): A list of Lean code snippets to verify.
            allow_sorry (bool, optional): Whether to allow 'sorry' in the code. Defaults to True.

        Returns:
            list[tuple[bool, list[dict]]]: A list of verification results, each containing:
                - bool: Whether the verification was successful.
                - list[dict]: Messages from the verification process.
        """
        if not codes:
            return []

        if not self.is_initialized:
            self.initialize_worker_pool()

        # start_time = time.time()

        # Submit all tasks and get futures
        tasks = [self.verify(code, allow_sorry) for code in codes]

        # Wait for all results
        results = await asyncio.gather(*tasks)

        return results

    async def extract_subgoals(self, code: str) -> list[str]:
        """
        Asynchronously extract subgoals from a piece of Lean code.

        This method submits the code to a worker process and asynchronously waits for the result.
        """
        if not self.is_initialized:
            self.initialize_worker_pool()

        task_id = self.next_task_id
        self.next_task_id += 1

        self.task_queue.put((task_id, code, "extract_subgoals"))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async def wait_for_result():
            while True:
                try:
                    result_task_id, result = self.result_queue.get_nowait()
                    if result_task_id == task_id:
                        future.set_result(result)
                        return

                    self.result_queue.put((result_task_id, result))
                except Exception:
                    await asyncio.sleep(0.01)

        asyncio.create_task(wait_for_result())

        return await future

    def __del__(self):
        """
        Ensure worker processes are shut down when the verifier is deleted.

        This destructor method ensures clean shutdown of worker processes when
        the AsyncVerifier instance is garbage collected.
        """
        self.shutdown()
