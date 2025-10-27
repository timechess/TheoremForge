from openai import AsyncOpenAI
from theoremforge.state import (
    TheoremForgeState,
    ProblemDecompositionTrace,
    SelfCorrectionTrace,
)
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
from theoremforge.utils import extract_lean_code, payload_to_string
from loguru import logger
from theoremforge.agents.informal_proof_agent import InformalProofAgent
from theoremforge.agents.verifier_agent import VerifierAgent
from theoremforge.agents.subgoal_extraction_agent import SubgoalExtractionAgent
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.agents.theorem_retriever_agent import TheoremRetrieverAgent
from theoremforge.agents.self_correction_agent import SelfCorrectionAgent
from typing import Tuple, List
import asyncio

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class DecompositionAgent(BaseAgent):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        theorem_retriever_agent: TheoremRetrieverAgent,
        informal_proof_agent: InformalProofAgent,
        verifier_agent: VerifierAgent,
        self_correction_agent: SelfCorrectionAgent,
        subgoal_extraction_agent: SubgoalExtractionAgent,
    ):
        super().__init__(
            agent_name="decomposition_agent",
            subagents=[
                theorem_retriever_agent,
                informal_proof_agent,
                verifier_agent,
                subgoal_extraction_agent,
                self_correction_agent,
            ],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def run(
        self, state: TheoremForgeState, formal_statement: str, **kwargs
    ) -> Tuple[TheoremForgeState, ProblemDecompositionTrace | None]:
        state, theorem_retriever_trace = await self.subagents[
            "theorem_retriever_agent"
        ].run(state, formal_statement)

        useful_theorems = theorem_retriever_trace.theorem_selection_results
        useful_theorems = "\n".join(
            [payload_to_string(theorem) for theorem in useful_theorems]
        )

        state, informal_proof_trace = await self.subagents["informal_proof_agent"].run(
            state, formal_statement, useful_theorems
        )
        if state.stage == "finished":
            return state, None

        informal_proofs = informal_proof_trace.informal_proof
        problem_decomposition_prompts = [
            env.get_template("problem_decomposition.j2").render(
                formal_statement=state.formal_statement,
                informal_proof=informal_proof,
                useful_theorems=useful_theorems,
            )
            for informal_proof in informal_proofs
        ]

        async def get_problem_decomposition_response(prompt: str) -> List[str]:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return [choice.message.content for choice in response.choices]

        problem_decomposition_responses = await asyncio.gather(
            *[
                get_problem_decomposition_response(prompt)
                for prompt in problem_decomposition_prompts
            ]
        )
        problem_decomposition_codes = [
            [
                extract_lean_code(content)
                for content in response
                if extract_lean_code(content)
            ]
            for response in problem_decomposition_responses
        ]
        if not any(sum(problem_decomposition_codes, [])):
            logger.info(
                "Decomposition Agent: Code generation failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
            return state, None
        proof_sketches = []
        for codes in problem_decomposition_codes:
            for code in codes:
                if not code:
                    continue
                state, verifier_trace = await self.subagents["verifier_agent"].run(
                    state, code, True
                )

                if verifier_trace.valid:
                    proof_sketches.append(code)
                else:
                    self_correction_prompt = env.get_template(
                        "self_correction.j2"
                    ).render(
                        incorrect_code=code,
                        error_message=verifier_trace.error_str,
                    )
                    self_correction_response = (
                        await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": self_correction_prompt}
                            ],
                            **kwargs,
                        )
                    )
                    self_correction_code = extract_lean_code(
                        self_correction_response.choices[0].message.content
                    )
                    self_correction_trace = SelfCorrectionTrace(
                        step="self_correction",
                        agent_name="self_correction_agent",
                        prompt=self_correction_prompt,
                        output=self_correction_response.choices[0].message.content,
                        output_code=self_correction_code,
                    )
                    state.trace.append(self_correction_trace)
                    state, verifier_trace = await self.subagents["verifier_agent"].run(
                        state, self_correction_code, True
                    )
                    if verifier_trace.valid:
                        proof_sketches.append(self_correction_code)
                    else:
                        continue

        trace = ProblemDecompositionTrace(
            step="problem_decomposition",
            agent_name="decomposition_agent",
            prompt=problem_decomposition_prompts,
            output=problem_decomposition_responses,
            formal_statement=formal_statement,
            informal_proof=informal_proofs,
            proof_sketch=proof_sketches,
        )
        state.trace.append(trace)
        if not proof_sketches:
            logger.info(
                "Decomposition Agent: Proof sketch generation failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
            return state, None
        state.proof_sketch = proof_sketches
        state, subgoal_extraction_trace = await self.subagents[
            "subgoal_extraction_agent"
        ].run(state, proof_sketches)
        return state, trace
