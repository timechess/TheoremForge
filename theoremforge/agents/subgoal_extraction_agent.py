from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from loguru import logger
from uuid import uuid4
from theoremforge.prompt_manager import prompt_manager
import re
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.utils import call_llm_interruptible

class SubgoalExtractionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="subgoal_extraction_agent", context=context)
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_subgoals(self, proof_sketch: str) -> list[str]:
        pattern = r"<subgoal>(.*?)</subgoal>"
        matches = re.findall(pattern, proof_sketch, re.DOTALL)
        return [match.strip() for match in matches]

    async def _run(self, state: TheoremForgeState):
        if not self.context.use_extract_goal:
            prompt = prompt_manager.subgoal_extraction(state.proof_sketch)
            response = await call_llm_interruptible(
                state,
                self.context,
                self.client,
                self.model_name,
                prompt,
                self.sampling_params,
                "subgoal_extraction_agent",
            )
            output = response[0]
            subgoals = self._extract_subgoals(output)
        else:
            subgoals = await self.context.verifier.extract_subgoals(state.proof_sketch)
        subgoal_ids = [str(uuid4()) for _ in subgoals]

        if not subgoals:
            logger.debug(
                f"Subgoal Extraction Agent: No subgoals found for state {state.id}"
            )
            await self.add_state_request("finish_agent", state)
            return
        logger.debug(
            f"Subgoal Extraction Agent: Found {len(subgoals)} subgoals for state {state.id}"
        )
        for subgoal_id, subgoal in zip(subgoal_ids, subgoals):
            new_state = TheoremForgeState(
                id=subgoal_id,
                formal_statement=subgoal,
                parent_id=state.id,
                siblings=subgoal_ids,
                depth=state.depth + 1,
                metadata={
                    "useful_theorems": state.metadata["useful_theorems"],
                },
            )
            await self.register_cancellation_event(new_state)
            await self.add_state_request("prover_agent", new_state)
        state.subgoals = subgoal_ids
        await self.add_state_request("proof_assembly_agent", state)
        
