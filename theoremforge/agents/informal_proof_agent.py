from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeState, InformalProofTrace
from typing import Tuple
from openai import AsyncOpenAI
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
import re
from loguru import logger

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class InformalProofAgent(BaseAgent):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        super().__init__(
            agent_name="informal_proof_agent",
            subagents=[],
        )

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def _extract_informal_proof(self, text: str) -> str | None:
        pattern = r"<informal_proof>(.*?)</informal_proof>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    async def run(
        self, state: TheoremForgeState, formal_statement: str, useful_theorems: str, **kwargs
    ) -> Tuple[TheoremForgeState, InformalProofTrace]:
        prompt = env.get_template("informal_proof_generation.j2").render(
            formal_statement=formal_statement, useful_theorems=useful_theorems
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        informal_proofs = [
            self._extract_informal_proof(choice.message.content)
            for choice in response.choices
            if self._extract_informal_proof(choice.message.content)
        ]
        trace = InformalProofTrace(
            step="informal_proof_generation",
            agent_name="informal_proof_agent",
            prompt=prompt,
            output=[choice.message.content for choice in response.choices],
            formal_statement=formal_statement,
            informal_proof=informal_proofs,
        )
        state.trace.append(trace)
        if not any(informal_proofs):
            logger.info(
                "Informative Proof Agent: Informal proof generation failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
            return state, trace
        return state, trace
