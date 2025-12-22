from theoremforge.agents.base_agent import BaseAgent
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from theoremforge.prompt_manager import prompt_manager
from loguru import logger
from theoremforge.utils import extract_lean_code, call_llm_interruptible
import asyncio


class ProofAssemblyAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="proof_assembly_agent",
            context=context,
        )
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        # Check if subgoals are ready in record
        async with self.context.record_lock:
            subgoals_in_record = all(
                [
                    subgoal_id in self.context.proof_record
                    for subgoal_id in state.subgoals
                ]
            )
            all_subgoals_ready = False
            subgoal_proofs = []

            if subgoals_in_record:
                all_subgoals_ready = all(
                    [
                        subgoal_id in self.context.proof_record
                        and self.context.proof_record[subgoal_id]
                        for subgoal_id in state.subgoals
                    ]
                )
                if all_subgoals_ready:
                    subgoal_proofs = [
                        self.context.proof_record[subgoal_id]
                        for subgoal_id in state.subgoals
                    ]
                    subgoal_statements = [
                        self.context.statement_record[subgoal_id]
                        for subgoal_id in state.subgoals
                    ]

        if subgoals_in_record:
            if all_subgoals_ready:
                logger.info(
                    f"Proof Assembly Agent: All subgoals found for state {state.id}"
                )

                prompt = prompt_manager.proof_assembly(
                    state.formal_statement,
                    state.proof_sketch,
                    "\n".join(subgoal_statements),
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "proof_assembly_agent",
                )
                content = response[0]
                code = extract_lean_code(content)
                if code:
                    # Check for cancellation before verification
                    if await self.is_cancelled(state):
                        logger.info(
                            f"Proof Assembly Agent: State {state.id} cancelled before verification"
                        )
                        await self.add_state_request("finish_agent", state)
                        return

                    (
                        valid,
                        messages,
                        error_str,
                    ) = await self.context.verifier.verify(
                        "\n".join(subgoal_proofs + [code]), False
                    )
                    if valid:
                        state.formal_proof = "\n".join(subgoal_proofs + [code])
                        state.success = True
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                    else:
                        logger.info(
                            f"Proof Assembly Agent: Assembly failed for state {state.id}, routing to finish_agent"
                        )
                        state.metadata["failed_assembly"] = {
                            "code": code,
                            "error": error_str,
                        }
                        await self.add_state_request("assembly_correction_agent", state)

                else:
                    logger.info(
                        f"Proof Assembly Agent: Failed to generate formal proof for state {state.id}"
                    )

                    await self.add_state_request("finish_agent", state)

            else:
                logger.info(
                    f"Proof Assembly Agent: Some subgoals failed for state {state.id}"
                )
                await self.add_state_request("finish_agent", state)
        else:
            await self.task_queue.put(state)
            await asyncio.sleep(1)
