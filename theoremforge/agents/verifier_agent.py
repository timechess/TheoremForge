from theoremforge.agents.base_agent import BaseAgent
from theoremforge.lean_server.client import RemoteVerifier
from theoremforge.state import TheoremForgeState, VerifierTrace
from typing import Tuple


class VerifierAgent(BaseAgent):
    def __init__(self, verifier_url: str) -> None:
        super().__init__(
            agent_name="verifier_agent",
            subagents=[],
        )
        self.verifier = RemoteVerifier(verifier_url)

    async def run(
        self, state: TheoremForgeState, code: str, allow_sorry: bool = True, **kwargs
    ) -> Tuple[TheoremForgeState, VerifierTrace]:
        valid, messages, error_str = await self.verifier.verify(code, allow_sorry)
        trace = VerifierTrace(
            step="verification",
            agent_name="verifier_agent",
            code=code,
            messages=messages,
            valid=valid,
            error_str=error_str,
        )
        state.trace.append(trace)
        return state, trace
