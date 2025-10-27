from theoremforge.agents.base_agent import BaseAgent
from theoremforge.agents.verifier_agent import VerifierAgent
from openai import AsyncOpenAI
from theoremforge.state import TheoremForgeState, ProofAssemblyTrace
from typing import Tuple
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
from loguru import logger
from theoremforge.utils import extract_lean_code

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class ProofAssemblyAgent(BaseAgent):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        verifier_agent: VerifierAgent,
    ):
        super().__init__(
            agent_name="proof_assembly_agent",
            subagents=[verifier_agent],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def run(
        self, state: TheoremForgeState, **kwargs
    ) -> Tuple[TheoremForgeState, ProofAssemblyTrace]:
        subgoal_proofs = "\n".join([proof for proof in state.subgoal_proofs])
        proof_assembly_prompt = env.get_template("proof_assembly.j2").render(
            formal_statement=state.formal_statement,
            proof_sketch=state.proof_sketch[state.successful_id],
            subgoal_proofs=subgoal_proofs,
        )
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": proof_assembly_prompt},
            ],
            **kwargs,
        )
        content = response.choices[0].message.content
        code = extract_lean_code(content)
        trace = ProofAssemblyTrace(
            step="proof_assembly",
            agent_name="proof_assembly_agent",
            prompt=proof_assembly_prompt,
            output=content,
            formal_statement=state.formal_statement,
            proof_sketch=state.proof_sketch[state.successful_id],
            subgoal_proofs=subgoal_proofs,
            final_proof=code,
        )
        state.trace.append(trace)
        if not code:
            logger.info(
                "Proof Assembly Agent: Code generation failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
            return state, trace
        state, verifier_trace = await self.subagents["verifier_agent"].run(state, code, False)
        if verifier_trace.valid:
            state.formal_proof = code
            state.result = "success"
            state.stage = "finished"
        else:
            logger.info(
                "Proof Assembly Agent: Code verification failed. Moving to finished."
            )
            state.stage = "finished"
            state.result = "failure"
        return state, trace
