from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeState, SelfCorrectionTrace
from typing import Tuple
from openai import AsyncOpenAI
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
from theoremforge.utils import extract_lean_code

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))

class SelfCorrectionAgent(BaseAgent):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        super().__init__(
            agent_name="self_correction_agent",
            subagents=[],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def run(
        self,
        state: TheoremForgeState,
        incorrect_code: str,
        error_message: str,
        **kwargs,
    ) -> Tuple[TheoremForgeState, SelfCorrectionTrace]:
        prompt = env.get_template("self_correction.j2").render(
            incorrect_code=incorrect_code,
            error_message=error_message,
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        code = extract_lean_code(content)
        trace = SelfCorrectionTrace(
            step="self_correction",
            agent_name="self_correction_agent",
            prompt=prompt,
            output=content,
            output_code=code,
        )
        state.trace.append(trace)
        return state, trace