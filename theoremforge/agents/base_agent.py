from typing import List, Tuple
from theoremforge.state import TheoremForgeState, BaseTrace
from abc import abstractmethod

class BaseAgent:
    def __init__(self, agent_name: str, subagents: List["BaseAgent"], *args, **kwargs):
        self.agent_name = agent_name
        self.subagents = {subagent.agent_name: subagent for subagent in subagents}

    @abstractmethod
    async def run(self, state: TheoremForgeState, **kwargs) -> Tuple[TheoremForgeState, BaseTrace]:
        raise NotImplementedError