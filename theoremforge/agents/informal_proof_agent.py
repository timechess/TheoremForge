from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from theoremforge.prompt_manager import prompt_manager
from openai import AsyncOpenAI
import re
from google.genai import Client
from theoremforge.utils import call_llm_interruptible


class InformalProofAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="informal_proof_agent",
            context=context,
        )
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_informal_proof(self, text: str) -> str | None:
        pattern = r"<informal_proof>(.*?)</informal_proof>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    async def _run(self, state: TheoremForgeState):
        prompt = prompt_manager.informal_proof_generation(
            state.formal_statement,
            state.metadata["useful_theorems"],
        )
        response = await call_llm_interruptible(
            state,
            self.context,
            self.client,
            self.model_name,
            prompt,
            self.sampling_params,
            "informal_proof_agent",
        )
        informal_proof = self._extract_informal_proof(response[0])

        state.informal_proof = informal_proof
        if state.parent_id:
            await self.add_state_request("shallow_solve_agent", state)
        else:
            await self.add_state_request("proof_sketch_agent", state)
