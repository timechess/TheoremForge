from theoremforge.state import TheoremForgeState, TheoremForgeContext
from abc import abstractmethod
import asyncio
from loguru import logger
from theoremforge.db import MongoDBClient


class BaseAgent:
    def __init__(self, agent_name: str, context: TheoremForgeContext, *args, **kwargs):
        self.agent_name = agent_name
        self.task_queue: asyncio.Queue[TheoremForgeState] = asyncio.Queue()
        self.context = context

    @property
    def db(self) -> MongoDBClient:
        """Access shared database client from context."""
        return self.context.db

    @abstractmethod
    async def run(self):
        raise NotImplementedError

    async def add_state_request(self, agent_name: str, state: TheoremForgeState):
        task_queue_lengths = [
            agent.task_queue.qsize() for agent in self.context.agents[agent_name]
        ]
        min_index = task_queue_lengths.index(min(task_queue_lengths))
        logger.debug(f"Adding state {state.id} to agent {agent_name} queue")
        await self.context.agents[agent_name][min_index].task_queue.put(state)
