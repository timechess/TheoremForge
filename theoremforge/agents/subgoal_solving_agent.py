from hashlib import sha256
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeState, SubgoalSolvingTrace
from typing import Tuple
from openai import AsyncOpenAI
from theoremforge.async_job_pool import AsyncJobPool, JobExecutionMode
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
from theoremforge.agents.prover_agent import ProverAgent

env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))


class SubgoalSolvingAgent(BaseAgent):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        prover_agent: ProverAgent,
    ):
        super().__init__(
            agent_name="subgoal_solving_agent",
            subagents=[prover_agent],
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    async def run(
        self, state: TheoremForgeState, id: int, **kwargs
    ) -> Tuple[TheoremForgeState, SubgoalSolvingTrace]:
        job_pool = AsyncJobPool()
        subgoals = state.subgoals[id]

        async def solve_subgoals(state: TheoremForgeState, subgoal: str):
            temp_state = TheoremForgeState(
                id=sha256(subgoal.encode("utf-8")).hexdigest()[:12],
                formal_statement=subgoal,
                stage="first_attempt",
                result="not_finished",
                trace=[],
            )
            temp_state, prover_trace = await self.subagents["prover_agent"].run(
                temp_state, subgoal, **kwargs
            )
            state.trace.extend(temp_state.trace)
            if not temp_state.formal_proof:
                raise ValueError("No valid proof found")
            return temp_state.formal_proof

        results = await job_pool.execute_jobs(
            solve_subgoals,
            [(state, subgoal) for subgoal in subgoals],
            JobExecutionMode.FIRST_FAILURE,
        )
        if not results["exceptions"]:
            subgoal_proofs = results["results"]
            state.subgoal_proofs = subgoal_proofs
            state.stage = "proof_assembly"
            trace = SubgoalSolvingTrace(
                step="subgoal_solving",
                agent_name="subgoal_solving_agent",
                formal_statements=subgoals,
                formal_proofs=subgoal_proofs,
            )
            state.trace.append(trace)
            state.successful_id = id
            return state, trace
        else:
            trace = SubgoalSolvingTrace(
                step="subgoal_solving",
                agent_name="subgoal_solving_agent",
                formal_statements=subgoals,
                formal_proofs=None,
            )
            state.trace.append(trace)
            return state, trace
