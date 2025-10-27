from openai import AsyncOpenAI
from theoremforge.state import TheoremForgeState, ProverTrace
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
from loguru import logger
from theoremforge.utils import extract_lean_code
from theoremforge.agents.base_agent import BaseAgent
from typing import Tuple
from theoremforge.agents.verifier_agent import VerifierAgent
from theoremforge.agents.self_correction_agent import SelfCorrectionAgent
import asyncio

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class ProverAgent(BaseAgent):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        verifier_agent: VerifierAgent,
        self_correction_agent: SelfCorrectionAgent,
    ) -> None:
        super().__init__(
            agent_name="prover_agent",
            subagents=[verifier_agent, self_correction_agent],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def run(
        self, state: TheoremForgeState, formal_statement: str, **kwargs
    ) -> Tuple[TheoremForgeState, ProverTrace]:
        prompt = env.get_template("proof_attempt.j2").render(
            formal_statement=formal_statement
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )
        codes = [
            extract_lean_code(choice.message.content) for choice in response.choices
        ]
        trace = ProverTrace(
            step="first_attempt",
            agent_name="prover_agent",
            prompt=prompt,
            output=[choice.message.content for choice in response.choices],
            formal_statement=formal_statement,
            output_code=codes,
        )
        state.trace.append(trace)
        valid_flag = False
        failed_code = []
        for code in codes:
            if not code:
                continue
            state, verifier_trace = await self.subagents["verifier_agent"].run(
                state, code, False
            )
            if verifier_trace.valid:
                valid_flag = True
                state.formal_proof = code
                state.result = "success"
                state.stage = "finished"
            else:
                failed_code.append((code, verifier_trace.error_str))

        if not valid_flag:

            async def self_correction_handler(
                state: TheoremForgeState, code: str, error_str: str
            ):
                state, self_correction_trace = await self.subagents[
                    "self_correction_agent"
                ].run(state, code, error_str, max_tokens=8192)
                if not self_correction_trace.output_code:
                    return state, None
                state, verifier_trace = await self.subagents["verifier_agent"].run(
                    state, self_correction_trace.output_code, False
                )
                return state, verifier_trace

            jobs = [
                asyncio.create_task(self_correction_handler(state, code, error_str))
                for code, error_str in failed_code
            ]
            results = await asyncio.gather(*jobs)
            for result in results:
                state, verifier_trace = result
                if not verifier_trace:
                    continue
                if verifier_trace.valid:
                    valid_flag = True
                    state.formal_proof = verifier_trace.code
                    state.result = "success"
                    state.stage = "finished"
                    break

        if not valid_flag:
            logger.info(
                "Prover Agent: First attempt failed. Moving to problem decomposition."
            )
            state.stage = "problem_decoposition"
        else:
            logger.info("Prover Agent: First attempt successful")
        return state, trace
