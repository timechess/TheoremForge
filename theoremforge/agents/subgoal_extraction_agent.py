from theoremforge.agents.base_agent import BaseAgent
from theoremforge.lean_server.client import RemoteVerifier
from theoremforge.state import TheoremForgeState, SubgoalExtractionTrace
from typing import Tuple, List
from loguru import logger


class SubgoalExtractionAgent(BaseAgent):
    def __init__(self, verifier_url: str):
        super().__init__(
            agent_name="subgoal_extraction_agent",
            subagents=[],
        )
        self.verifier = RemoteVerifier(verifier_url)

    async def run(
        self, state: TheoremForgeState, codes: List[str], **kwargs
    ) -> Tuple[TheoremForgeState, SubgoalExtractionTrace]:
        all_subgoals = []
        for i, code in enumerate(codes):
            subgoals = await self.verifier.extract_subgoals(code)
            if not subgoals:
                state.proof_sketch.pop(i)
                logger.info(
                    f"Subgoal Extraction Agent: Subgoal extraction failed for code {i}. Removing proof sketch."
                )
                continue
            all_subgoals.append(subgoals)
        trace = SubgoalExtractionTrace(
            step="subgoal_extraction",
            agent_name="subgoal_extraction_agent",
            proof_sketch=codes,
            subgoals=all_subgoals,
        )
        state.trace.append(trace)
        state.subgoals = all_subgoals
        if not all_subgoals:
            logger.info(
                "Subgoal Extraction Agent: Subgoal extraction failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
            return state, None
        logger.info(
            f"Subgoal Extraction Agent: Subgoal extraction successful. Extracted {len(all_subgoals)} subgoals."
        )
        state.stage = "subgoal_solving"
        return state, trace
